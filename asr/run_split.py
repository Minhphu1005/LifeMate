"""Chạy ASR cho cả một split, lưu transcript + thống kê sức khỏe.

  python -m asr.run_split --data "train+public_test" --split train --model vinai/PhoWhisper-medium --out asr_train.jsonl
  python -m asr.run_split --data "train+public_test" --split public_test --model vinai/PhoWhisper-medium --out asr_public_test.jsonl

Mỗi dòng output: {"id", "text", "history": [...], "sec", "n_audio"}.
Có thể chạy tiếp nếu bị ngắt (bỏ qua id đã có trong file --out).
Với train, nếu có audio_text thì chỉ dùng để in WER tổng cuối cùng (không đưa vào output).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .eval_asr import _edit, _norm
from .models.asr_factory import build_asr
from .pipeline import transcribe_sample


def suspicious(text: str) -> str:
    """Gắn cờ transcript đáng ngờ: rỗng, quá dài, lặp từ."""
    words = text.split()
    if not words:
        return "empty"
    if len(words) > 60:
        return "too_long"
    for n in (1, 2, 3):
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if len(grams) >= 6 and any(grams.count(g) >= 5 for g in set(grams)):
            return "repeat"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True, choices=["train", "public_test", "private_test"])
    ap.add_argument("--engine", default="phowhisper")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = tất cả")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.data)
    if not (root / a.split).exists() and (root / root.name / a.split).exists():
        root = root / root.name
    sdir = root / a.split
    rows = [json.loads(l) for l in open(sdir / f"{a.split}.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    asr = build_asr({"engine": a.engine, "model_path": a.model})

    flags, secs, we, wn = [], [], 0, 0
    with open(a.out, "a", encoding="utf-8") as fo:
        for k, item in enumerate(rows):
            if item["id"] in done:
                continue
            t0 = time.time()
            try:
                o = transcribe_sample(asr, item, sdir)
            except Exception as e:  # không để 1 mẫu làm sập cả lượt chạy
                o = {"text": "", "history": [], "error": repr(e)}
            sec = time.time() - t0
            n_audio = 1 + sum(1 for t in item.get("history", []) if t.get("role") == "user")
            flag = suspicious(o["text"]) or ("error" if "error" in o else "")
            if flag:
                flags.append((item["id"], flag))
            secs.append(sec)
            if item.get("audio_text"):
                r, h = _norm(item["audio_text"]).split(), _norm(o["text"]).split()
                we += _edit(r, h); wn += len(r)
            fo.write(json.dumps({"id": item["id"], **o, "sec": round(sec, 2), "n_audio": n_audio,
                                 "flag": flag}, ensure_ascii=False) + "\n")
            fo.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(rows)}  avg {sum(secs) / len(secs):.2f}s/sample", flush=True)

    print(f"xong {len(secs)} mẫu mới | avg {sum(secs) / max(len(secs), 1):.2f}s | max {max(secs, default=0):.2f}s")
    if wn:
        print(f"WER câu cuối (chỉ train): {we / wn:.4f}")
    print(f"mẫu bị gắn cờ: {len(flags)}", flags[:20])


if __name__ == "__main__":
    main()

