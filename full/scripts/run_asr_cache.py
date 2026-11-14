"""Chạy ASR trên toàn bộ một split và lưu cache jsonl (id, text, history, sec, n_audio).

Ví dụ:
  python scripts/run_asr_cache.py --data ../train+public_test --split train \
      --out sft_data/asr_train.jsonl --allow-download
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.asr.pipeline import transcribe_sample  # noqa: E402
from lifemate.data import locate_split, read_jsonl  # noqa: E402
from lifemate.runner import build_asr_from_cfg, load_config, seed_everything  # noqa: E402
from training.train_asr import edit, norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    ap.add_argument("--model", default=None, help="Ghi đè asr.model_path trong config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-chitchat-history", action="store_true",
                    help="Bỏ qua lượt user history xã giao (mặc định: phiên âm toàn bộ các lượt history)")
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seed_everything(cfg.get("system", {}).get("seed", 42))
    if a.model:
        cfg["asr"]["model_path"] = a.model
    split, sdir, _ = locate_split(a.data, a.split)
    rows = list(read_jsonl(sdir / f"{split}.jsonl"))
    if a.limit:
        rows = rows[: a.limit]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    asr = build_asr_from_cfg(cfg, a.allow_download)

    we = wn = 0
    secs = []
    with open(out_path, "a", encoding="utf-8") as fo:
        for k, item in enumerate(rows):
            if item["id"] in done:
                continue
            t0 = time.time()
            try:
                tr = transcribe_sample(asr, item, sdir, skip_chitchat_history=a.skip_chitchat_history)
            except Exception as e:
                tr = {"text": "", "history": [], "error": repr(e)}
            sec = time.time() - t0
            secs.append(sec)
            if item.get("audio_text"):
                r, h = norm(item["audio_text"]).split(), norm(tr["text"]).split()
                we += edit(r, h)
                wn += len(r)
            n_audio = 1 + sum(1 for h in item.get("history", []) if h.get("role") == "user")
            fo.write(json.dumps({"id": item["id"], **tr, "sec": round(sec, 2), "n_audio": n_audio},
                                ensure_ascii=False) + "\n")
            fo.flush()
            if (k + 1) % 50 == 0:
                print(f"[asr] {k + 1}/{len(rows)} avg={sum(secs) / len(secs):.2f}s", flush=True)
    print(f"[asr] xong {len(secs)} mẫu mới -> {out_path}")
    if wn:
        print(f"[asr] WER câu cuối trên {split}: {we / wn:.4f}")


if __name__ == "__main__":
    main()

