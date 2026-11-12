"""Chạy toàn bộ pipeline: audio -> ASR -> agent (LLM + ToolEnv) -> submission.jsonl."""
from __future__ import annotations

import csv
import json
import platform
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .agent.agent import run_agent
from .agent.prompt import Catalog
from .agent.retriever import FewShotRetriever, format_few_shot_block
from .data import import_toolenv, load_catalog_raw, locate_split, read_jsonl


def load_config(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int):
    import random
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def gold_transcripts(item: dict) -> dict:
    """CHỈ DÙNG KHI PHÁT TRIỂN trên train (đánh giá riêng phần LLM). Không bao giờ dùng khi nộp."""
    hist = []
    for t in item.get("history", []):
        if t.get("role") == "user":
            hist.append({"role": "user", "text": t.get("audio_text", "")})
        else:
            hist.append({"role": t.get("role", "assistant"), "text": t.get("text", ""),
                         "tool_calls": t.get("tool_calls", []), "tool_results": t.get("tool_results", [])})
    return {"text": item["audio_text"], "history": hist}


class _Timeout(Exception):
    pass


def _arm(seconds: float):
    if hasattr(signal, "SIGALRM") and seconds > 0:
        def handler(signum, frame):
            raise _Timeout()
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)


def _disarm():
    if hasattr(signal, "SIGALRM"):
        signal.setitimer(signal.ITIMER_REAL, 0)


def build_asr_from_cfg(cfg: dict, allow_download: bool):
    from .asr.models.asr_factory import build_asr
    from .agent.llm import resolve_model
    a = dict(cfg["asr"])
    if a.get("engine", "whisper") != "dummy":
        a["model_path"] = resolve_model(a["model_path"], a.get("hub_id"), allow_download)
    return build_asr(a)


def build_llm_from_cfg(cfg: dict, allow_download: bool):
    from .agent.llm import HFChatLLM, resolve_model
    l = dict(cfg["llm"])
    base = resolve_model(l.pop("base_model_path"), l.pop("hub_id", None), allow_download)
    adapter = l.pop("adapter_path", None)
    adapter = adapter if (adapter and Path(adapter).exists()) else None
    if l.pop("require_adapter", False) and adapter is None:
        raise FileNotFoundError("Chưa có LoRA adapter của LLM (llm.adapter_path) — chạy training/train_llm.py trước.")
    l.pop("engine", None)
    return HFChatLLM(base, adapter, seed=cfg.get("system", {}).get("seed", 42), **l)


def run(cfg: dict, data_dir: str, out_path: str, split: Optional[str] = None, limit: int = 0,
        text_source: str = "asr", asr_cache: Optional[str] = None, reports_dir: Optional[str] = None,
        allow_download: bool = False, llm=None, asr=None) -> dict:
    sysc = cfg.get("system", {})
    seed_everything(sysc.get("seed", 42))
    split, split_dir, root = locate_split(data_dir, split)
    ToolEnv = import_toolenv(split_dir)
    catalog = Catalog(load_catalog_raw(split_dir, split))
    items = list(read_jsonl(split_dir / f"{split}.jsonl"))
    if limit:
        items = items[:limit]
    print(f"[run] split={split} samples={len(items)} text_source={text_source}", flush=True)

    cache = {}
    if asr_cache:
        cache = {r["id"]: r for r in read_jsonl(Path(asr_cache))}
    if text_source == "asr" and not cache and asr is None:
        asr = build_asr_from_cfg(cfg, allow_download)
    if llm is None:
        llm = build_llm_from_cfg(cfg, allow_download)

    few_shot_k = int(sysc.get("few_shot_k", 2))
    retriever = None
    if few_shot_k > 0:
        bank_path = Path(sysc.get("few_shot_bank", "weights/few_shot_bank.jsonl"))
        if bank_path.is_file():
            retriever = FewShotRetriever.from_jsonl(bank_path)
        else:
            try:
                _, tr_dir, _ = locate_split(data_dir, "train")
                retriever = FewShotRetriever.from_jsonl(tr_dir / "train.jsonl")
            except Exception:
                retriever = None

    budget = float(sysc.get("time_budget_s", 25))
    rdir = Path(reports_dir) if reports_dir else Path(out_path).parent / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lat_file = open(rdir / "latency.csv", "w", newline="", encoding="utf-8")
    lat = csv.writer(lat_file)
    lat.writerow(["id", "asr_s", "agent_s", "total_s", "stop", "n_calls", "error"])
    fail = open(rdir / "failures.jsonl", "w", encoding="utf-8")
    with open(rdir / "env_info.txt", "w", encoding="utf-8") as f:
        f.write(f"python {sys.version}\nplatform {platform.platform()}\nseed {sysc.get('seed', 42)}\n")
        for m in ("torch", "transformers", "peft"):
            try:
                f.write(f"{m} {__import__(m).__version__}\n")
            except Exception:
                pass

    totals = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for k, item in enumerate(items):
            t0 = time.time()
            env = ToolEnv(split, item["id"], root=str(root))   # đúng MỘT ToolEnv cho mỗi sample
            asr_s = agent_s = 0.0
            stop, err = "", ""
            try:
                _arm(budget + 4)  # chốt chặn cứng cho cả sample (Linux)
                ta = time.time()
                if text_source == "gold":
                    tr = gold_transcripts(item)
                elif item["id"] in cache:
                    tr = cache[item["id"]]
                else:
                    from .asr.pipeline import transcribe_sample
                    asr_cfg = cfg.get("asr", {})
                    tr = transcribe_sample(
                        asr, item, split_dir,
                        skip_chitchat_history=asr_cfg.get("skip_chitchat_history", True),
                        use_context_prompt=asr_cfg.get("use_context_prompt", False),
                    )
                asr_s = time.time() - ta
                tb = time.time()
                fs_block = ""
                if retriever is not None and few_shot_k > 0:
                    exs = retriever.retrieve(
                        tr["text"], item["context"].get("installed_apps", {}),
                        k=few_shot_k, exclude_id=item["id"]
                    )
                    fs_block = format_few_shot_block(exs)
                info = run_agent(
                    llm, env, catalog, item["context"], tr["text"], tr.get("history", []),
                    max_steps=sysc.get("max_steps", 8), max_repairs=sysc.get("max_repairs", 2),
                    deadline=t0 + budget, few_shot_block=fs_block,
                    enable_xref_guard=sysc.get("enable_xref_guard", True),
                )
                agent_s = time.time() - tb
                stop = info["stop"]
            except BaseException as e:  # không để một sample làm sập cả lượt chạy; vẫn ghi đủ 1 dòng
                if isinstance(e, KeyboardInterrupt):
                    raise
                err = repr(e)
                stop = "timeout" if isinstance(e, _Timeout) else "error"
                fail.write(json.dumps({"id": item["id"], "error": err,
                                       "trace": traceback.format_exc()[-1500:]}, ensure_ascii=False) + "\n")
                fail.flush()
            finally:
                _disarm()
            calls = env.get_tool_calls()
            fout.write(json.dumps({"id": item["id"], "tool_calls": calls, "state_hash": env.get_state_hash()},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            total = time.time() - t0
            totals.append(total)
            lat.writerow([item["id"], f"{asr_s:.2f}", f"{agent_s:.2f}", f"{total:.2f}", stop, len(calls), err])
            if (k + 1) % 20 == 0:
                print(f"[run] {k + 1}/{len(items)} avg {sum(totals) / len(totals):.2f}s/sample", flush=True)
    lat_file.close()
    fail.close()
    print(f"[run] xong {len(items)} mẫu | avg {sum(totals) / max(len(totals), 1):.2f}s | max {max(totals, default=0):.2f}s")
    return {"split": split, "n": len(items), "avg_s": sum(totals) / max(len(totals), 1)}

