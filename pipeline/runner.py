import sys
import json
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

from .stubs import *
from .catalog import *

def set_seed(seed: int):
    """Cố định seed cho random, numpy, torch (tất định)."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def build_components(cfg: dict):
    """Xây dựng ASR và LLM qua factory, fallback sang DummyLLM nếu chưa có."""
    asr = None
    try:
        from asr.models.asr_factory import build_asr
        asr = build_asr(cfg.get("asr", {}))
    except Exception:
        try:
            from asr.models.asr_factory import build_asr
            asr = build_asr({"engine": "dummy"})
        except Exception:
            pass

    llm = None
    llm_cfg = cfg.get("llm", {})
    engine = llm_cfg.get("engine", "dummy")

    if engine != "dummy":
        try:
            from llm.models.llm_factory import build_llm
            llm = build_llm(llm_cfg)
        except Exception:
            llm = None

    if llm is None:
        llm = DummyLLM()

    return asr, llm

def canonical_json(args: dict) -> str:
    """Tạo chuỗi JSON chuẩn hóa để so khớp multiset."""
    return json.dumps(args, sort_keys=True, ensure_ascii=False)

def remaining_calls(calls: list, executed: list) -> list:
    """Khử trùng theo multiset (name + canonical JSON arguments)."""
    executed_multiset = {}
    for c in executed:
        key = (c["name"], canonical_json(c.get("arguments", {})))
        executed_multiset[key] = executed_multiset.get(key, 0) + 1

    filtered = []
    for c in calls:
        key = (c["name"], canonical_json(c.get("arguments", {})))
        if executed_multiset.get(key, 0) > 0:
            executed_multiset[key] -= 1
        else:
            filtered.append(c)
    return filtered

def filter_tools(catalog: dict, installed_apps: dict) -> list:
    """Lọc catalog theo installed_apps và luôn kèm sys.ask_missing_info[cite: 1, 2]."""
    names = {n for app in installed_apps.values() for n in app.get("tools", [])}
    available = [catalog[n] for n in names if n in catalog]
    if "sys.ask_missing_info" in catalog:
        available.append(catalog["sys.ask_missing_info"])
    return available

def transcribe_sample(asr, item, split_dir):
    """Gọi ASR hoặc fallback audio_text khi dev cục bộ."""
    if asr is not None and hasattr(asr, "transcribe"):
        try:
            return asr.transcribe(item, split_dir)
        except Exception:
            pass
    text = item.get("audio_text", "")
    history = item.get("history", [])
    return {"text": text, "history": history}

def process_sample(item, split, root, catalog, asr, llm, cfg, failures_file):
    """Xử lý 1 sample với vòng lặp, timeout và ghi nhận metric."""
    sys.path.insert(0, str(root / split))
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        from zaic_tools import ToolEnv

    env = ToolEnv(split, item["id"], root=root)

    t0 = time.time()
    asr_t = 0.0
    llm_t = 0.0
    env_t = 0.0
    n_env_calls = 0
    n_llm_turns = 1

    executed = []
    failed_writes = set()
    max_env_calls = cfg.get("system", {}).get("max_env_calls", 12)
    allow_write = cfg.get("system", {}).get("allow_write_in_env_caller", False)

    def validator(name, args):
        return env.validate(name, args)

    def kind(name):
        tool_info = catalog.get(name, {})
        return tool_info.get("kind", "read")

    def env_caller(name, args):
        nonlocal n_env_calls
        if n_env_calls >= max_env_calls:
            return {"ok": False, "error": "limit"}

        k = kind(name)
        if k == "read":
            pass
        elif allow_write and k == "write":
            if name in failed_writes:
                raise PermissionError("cấm retry write đã fail[cite: 1]")
            if not env.validate(name, args).get("ok"):
                raise ValueError("write chưa pass validator[cite: 1]")
        else:
            raise PermissionError("env_caller không cho phép tool này[cite: 1]")

        n_env_calls += 1
        t_start = time.time()
        res = env.call(name, args)
        env_t += time.time() - t_start

        if k == "write" and not res.get("ok", True):
            failed_writes.add(name)
        executed.append({"name": name, "arguments": args})
        return res

    try:
        t_asr_start = time.time()
        a = transcribe_sample(asr, item, root / split)
        asr_t = time.time() - t_asr_start

        installed = item.get("context", {}).get("installed_apps", {})
        tools = filter_tools(catalog, installed)

        t_llm_start = time.time()
        calls = llm.generate_calls(
            a["text"],
            a["history"],
            item.get("context", {}),
            tools,
            env_caller=env_caller,
            validator=validator
        )
        llm_t = time.time() - t_llm_start

        remaining = remaining_calls(calls, executed)
        for c in remaining:
            if n_env_calls >= max_env_calls:
                break
            name = c["name"]
            args = c.get("arguments", {})
            t_start = time.time()
            n_env_calls += 1
            res = env.call(name, args)
            env_t += time.time() - t_start
            executed.append({"name": name, "arguments": args})

    except Exception as e:
        err_msg = str(e)
        stage = "timeout" if isinstance(e, TimeoutError) else "env"
        if "asr" in err_msg.lower():
            stage = "asr"
        elif "llm" in err_msg.lower() or isinstance(e, (AttributeError, TypeError)):
            stage = "llm"
        with open(failures_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], "stage": stage, "error": err_msg}, ensure_ascii=False) + "\n")

    total_s = time.time() - t0
    lat_record = {
        "id": item["id"],
        "asr_s": round(asr_t, 4),
        "llm_s": round(llm_t, 4),
        "env_s": round(env_t, 4),
        "total_s": round(total_s, 4),
        "n_env_calls": n_env_calls,
        "n_llm_turns": n_llm_turns
    }

    return {
        "id": item["id"],
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash()
    }, lat_record

def run(cfg: dict, split: str, data_dir, out, limit: int = 0):
    root = resolve_layout(data_dir, split)
    print(f"Found {split}.jsonl at {root}")

    catalog_path = root / split / f"{split}_tools_catalog.json"
    catalog = load_catalog(catalog_path)

    asr, llm = build_components(cfg)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.jsonl"

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures_file = reports_dir / "failures.jsonl"
    latency_file = reports_dir / "latency.csv"
    env_info_file = reports_dir / "env_info.txt"

    for p in [out_path, failures_file, latency_file, env_info_file]:
        if p.exists():
            p.unlink()

    with open(latency_file, "w", encoding="utf-8") as f_lat:
        f_lat.write("id,asr_s,llm_s,env_s,total_s,n_env_calls,n_llm_turns\n")

    jsonl_path = root / split / f"{split}.jsonl"
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if limit > 0:
        items = items[:limit]

    seed = cfg.get("system", {}).get("seed", 42)
    with open(env_info_file, "w", encoding="utf-8") as f_env:
        f_env.write(f"seed: {seed}\n")
        f_env.write(f"date: {datetime.now().isoformat()}\n")
        f_env.write(f"config: {json.dumps(cfg, ensure_ascii=False)}\n")

    set_seed(seed)

    with open(out_path, "w", encoding="utf-8") as f_out, \
         open(latency_file, "a", encoding="utf-8") as f_lat:

        for item in items:
            res, lat = process_sample(item, split, root, catalog, asr, llm, cfg, failures_file)
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            f_out.flush()

            f_lat.write(f"{lat['id']},{lat['asr_s']},{lat['llm_s']},{lat['env_s']},{lat['total_s']},{lat['n_env_calls']},{lat['n_llm_turns']}\n")
            f_lat.flush()

    print(f"Completed run for {split}. Output saved to {out_path}")

