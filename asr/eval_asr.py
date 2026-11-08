"""Đánh giá ASR trên train: WER / CER / thời gian, gồm cả audio history.

Ví dụ:
  python -m asr.eval_asr --data "train+public_test" \
      --engine whisper --model openai/whisper-small --limit 200 --out asr_eval_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from pathlib import Path

from .models.asr_factory import build_asr


def _norm(s: str) -> str:
    """Chuẩn hóa để chấm: NFC, chữ thường, bỏ dấu câu."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def collect(split_dir: Path, include_history: bool = True):
    items = []
    with open(split_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append((r["id"], split_dir / r["audio"], r["audio_text"]))
            if include_history:
                for h in r["history"]:
                    if h["role"] == "user":
                        items.append((Path(h["audio"]).stem, split_dir / h["audio"], h["audio_text"]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục chứa train/ (và public_test/)")
    ap.add_argument("--engine", default="whisper")
    ap.add_argument("--model", required=True, help="đường dẫn hoặc tên model HF")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="asr_eval.jsonl")
    a = ap.parse_args()

    root = Path(a.data)
    if not (root / "train").exists() and (root / root.name / "train").exists():
        root = root / root.name  # thư mục bị lồng: train+public_test/train+public_test/
    items = collect(root / "train", not a.no_history)
    random.Random(42).shuffle(items)
    items = items[: a.limit]

    asr = build_asr({"engine": a.engine, "model_path": a.model, "prompt": a.prompt,
                     "num_beams": a.beams, "batch_size": a.batch_size})
    t0 = time.time()
    hyps = asr.transcribe_batch([str(p) for _, p, _ in items])
    dt = time.time() - t0

    we = wn = ce = cn = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for (sid, _, ref), hyp in zip(items, hyps):
            r, h = _norm(ref).split(), _norm(hyp).split()
            we += _edit(r, h); wn += len(r)
            rc, hc = list(_norm(ref).replace(" ", "")), list(_norm(hyp).replace(" ", ""))
            ce += _edit(rc, hc); cn += len(rc)
            fo.write(json.dumps({"id": sid, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
    print(f"n={len(items)}  WER={we / wn:.4f}  CER={ce / cn:.4f}  "
          f"time/sample={dt / len(items):.2f}s  -> {a.out}")


if __name__ == "__main__":
    main()
