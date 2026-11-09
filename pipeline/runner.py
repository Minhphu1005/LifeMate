"""Chạy ASR -> LLM -> ToolEnv cho từng sample, ghi submission.jsonl + reports.

Quy tắc cứng (xem cong-viec-3-PHUC-pipeline.txt):
  * ĐÚNG 1 ToolEnv/sample; chỉ dùng validate/call/get_tool_calls/get_state_hash.
  * tool_calls nộp = env.get_tool_calls() (log của env), không phải list LLM trả.
  * env_caller (đưa cho LLM) chỉ cho read tool; write do pipeline thực thi sau,
    không retry. Call đã chạy qua env_caller không chạy lại (multiset).
  * Mỗi sample: try/except + timeout cứng; luôn ghi đủ 1 dòng.
"""
from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .catalog import ASK, filter_tools, load_catalog, resolve_layout


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def build_components(cfg: dict):
    """Nạp ASR + LLM qua factory; engine dummy có fallback sang stub nếu module chưa có."""
    from asr.models.asr_factory import build_asr
    asr = build_asr(cfg["asr"])
    try:
        from llm.models.llm_factory import build_llm
        llm = build_llm(cfg["llm"])
    except ImportError:
        if cfg["llm"].get("engine", "dummy") != "dummy":
            raise
        from .stubs import DummyLLM
        llm = DummyLLM()
    return asr, llm


def _key(call: dict) -> str:
    return call["name"] + "|" + json.dumps(call.get("arguments", {}), sort_keys=True,
                                           ensure_ascii=False)


def remaining_calls(calls: List[dict], executed: List[dict]) -> List[dict]:
    """calls trừ executed (multiset theo (name, canonical args)), giữ thứ tự của calls."""
    left = Counter(_key(c) for c in executed)
    out = []
    for c in calls:
        k = _key(c)
        if left[k] > 0:
            left[k] -= 1
        else:
            out.append(c)
    return out


class SampleRun:
    """Trạng thái 1 sample: bọc env cho LLM, theo dõi thời gian và call đã chạy."""

    def __init__(self, env, tools: List[dict], deadline: float, max_env_calls: int,
                 allow_write: bool = False):
        self.env = env
        self.kind = {t["name"]: t.get("kind") for t in tools}
        self.deadline = deadline
        self.max_env_calls = max_env_calls
        self.allow_write = allow_write
        self.failed_writes: set = set()
        self.executed: List[dict] = []      # call đã chạy qua env_caller
        self.n_env = 0
        self.env_s = 0.0
        self.cancelled = threading.Event()
        self.notes: List[dict] = []

    def _check(self):
        if self.cancelled.is_set() or time.time() > self.deadline:
            raise TimeoutError("sample timeout")

    def _env_call(self, name, args):
        self._check()
        if self.n_env >= self.max_env_calls:
            return {"ok": False, "error": "env call limit"}
        t0 = time.time()
        try:
            return self.env.call(name, args)
        finally:
            self.n_env += 1
            self.env_s += time.time() - t0

    def validator(self, name, args):
        self._check()
        return self.env.validate(name, args)        # dry-run, không ghi log

    def env_caller(self, name, args):
        """Mặc định chỉ read. Nếu allow_write=True: cho cả write đã pass validator, cấm retry."""
        k = self.kind.get(name)
        args = dict(args or {})
        if k == "read":
            pass
        elif self.allow_write and k == "write":
            if name in self.failed_writes:
                raise PermissionError(f"cấm retry write tool đã lỗi: {name}")
            v = self.validator(name, args)
            if isinstance(v, dict) and not v.get("ok", True):
                raise ValueError(f"write chưa pass validator: {v.get('error')}")
        else:
            raise PermissionError(f"env_caller không cho phép gọi tool: {name}")
        res = self._env_call(name, args)
        if not (isinstance(res, dict) and res.get("error") == "env call limit"):
            self.executed.append({"name": name, "arguments": args})
            if k == "write" and isinstance(res, dict) and not res.get("ok", True):
                self.failed_writes.add(name)
        return res

    def execute_rest(self, calls: List[dict]):
        """Thực thi phần còn lại (write / ask_missing_info / read chưa chạy). Không retry."""
        for c in remaining_calls(calls, self.executed):
            if not isinstance(c, dict) or "name" not in c or c["name"] not in self.kind:
                self.notes.append({"stage": "llm", "error": f"bỏ call không hợp lệ: {c!r}"[:300]})
                continue
            args = c.get("arguments")
            if not isinstance(args, dict):
                self.notes.append({"stage": "llm", "error": f"arguments không phải dict: {c!r}"[:300]})
                continue
            self._env_call(c["name"], args)


def process_sample(item: dict, split: str, root: Path, catalog: Dict[str, dict],
                   asr, llm, cfg: dict, ToolEnv, transcribe_sample) -> dict:
    sys_cfg = cfg.get("system", {})
    timeout = float(sys_cfg.get("timeout_per_sample", 25))
    max_env = int(sys_cfg.get("max_env_calls", 12))
    allow_write = bool(sys_cfg.get("allow_write_in_env_caller", False))
    t_start = time.time()
    env = ToolEnv(split, item["id"], root=root)         # ĐÚNG 1 lần/sample
    ctx = item.get("context") or {}
    tools, missing = filter_tools(catalog, ctx.get("installed_apps"))
    run = SampleRun(env, tools, t_start + timeout, max_env, allow_write=allow_write)
    stats = {"asr_s": 0.0, "llm_s": 0.0}
    failures: List[dict] = [{"stage": "catalog", "error": f"tool không có trong catalog: {m}"}
                            for m in missing]

    def work():
        t0 = time.time()
        a = transcribe_sample(asr, item, root / split)
        stats["asr_s"] = time.time() - t0
        t1 = time.time()
        calls = llm.generate_calls(a["text"], a["history"], ctx, tools,
                                   run.env_caller, run.validator)
        stats["llm_s"] = time.time() - t1 - run.env_s
        run.execute_rest(calls or [])

    err: List[str] = []

    def target():
        try:
            work()
        except TimeoutError as e:
            err.append(f"timeout: {e}")
        except Exception as e:                          # không để 1 mẫu làm sập pipeline
            err.append(repr(e))

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        run.cancelled.set()
        failures.append({"stage": "timeout", "error": f">{timeout}s"})
    if err:
        stage = "timeout" if err[0].startswith("timeout") else "pipeline"
        failures.append({"stage": stage, "error": err[0][:500]})
    failures.extend(run.notes)

    return {"record": {"id": item["id"], "tool_calls": env.get_tool_calls(),
                       "state_hash": env.get_state_hash()},
            "failures": failures,
            "latency": {"id": item["id"], "asr_s": round(stats["asr_s"], 3),
                        "llm_s": round(max(stats["llm_s"], 0.0), 3),
                        "env_s": round(run.env_s, 3),
                        "total_s": round(time.time() - t_start, 3),
                        "n_env_calls": run.n_env}}


def _write_env_info(path: Path, cfg: dict):
    info = [f"python {platform.python_version()}", f"platform {platform.platform()}",
            f"seed {cfg.get('system', {}).get('seed')}"]
    for mod in ("torch", "transformers", "numpy", "yaml"):
        try:
            m = __import__(mod)
            info.append(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            info.append(f"{mod} (không cài)")
    try:
        info.append("git " + subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip())
    except Exception:
        pass
    info.append("config " + json.dumps(cfg, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(info) + "\n", encoding="utf-8")


def run(cfg: dict, split: str, data_dir, out, limit: int = 0) -> Path:
    set_seed(int(cfg.get("system", {}).get("seed", 42)))
    root = resolve_layout(data_dir, split)
    split_dir = root / split
    sys.path.insert(0, str(split_dir))                  # zaic_tools của CHÍNH split này
    from zaic_tools import ToolEnv                      # type: ignore
    from asr.pipeline import transcribe_sample

    catalog = load_catalog(split_dir / f"{split}_tools_catalog.json")
    asr, llm = build_components(cfg)
    with open(split_dir / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if limit:
        rows = rows[:limit]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = Path(cfg.get("paths", {}).get("reports") or out.parent / "reports")
    rep.mkdir(parents=True, exist_ok=True)
    _write_env_info(rep / "env_info.txt", cfg)

    lat_fields = ["id", "asr_s", "llm_s", "env_s", "total_s", "n_env_calls"]
    with open(out, "w", encoding="utf-8") as fo, \
            open(rep / "latency.csv", "w", encoding="utf-8", newline="") as fl, \
            open(rep / "failures.jsonl", "w", encoding="utf-8") as ff:
        w = csv.DictWriter(fl, fieldnames=lat_fields)
        w.writeheader()
        for k, item in enumerate(rows):
            try:
                r = process_sample(item, split, root, catalog, asr, llm, cfg, ToolEnv,
                                   transcribe_sample)
            except Exception as e:      # lỗi ngay cả khi tạo env: vẫn phải có 1 dòng
                r = {"record": {"id": item["id"], "tool_calls": [], "state_hash": ""},
                     "failures": [{"stage": "env_init", "error": repr(e)[:500]}],
                     "latency": {k2: 0 for k2 in lat_fields} | {"id": item["id"]}}
            fo.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
            fo.flush()
            w.writerow(r["latency"])
            for f in r["failures"]:
                ff.write(json.dumps({"id": item["id"], **f}, ensure_ascii=False) + "\n")
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(rows)}", flush=True)
    print(f"xong {len(rows)} mẫu -> {out}")
    return out
