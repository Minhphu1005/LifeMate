"""Entrypoint: python run_pipeline.py --config config.yaml

Docker: đọc /data, ghi /result/submission.jsonl (xem Dockerfile).
Phát triển: thêm --split train --limit 100 --text-source gold|asr --asr-cache asr_train.jsonl ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lifemate.runner import load_config, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "/data"))
    ap.add_argument("--out", default=os.environ.get("OUT_FILE", "/result/submission.jsonl"))
    ap.add_argument("--split", default=None, help="mặc định tự dò trong --data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--text-source", choices=["asr", "gold"], default="asr",
                    help="gold = dùng transcript chuẩn của train (CHỈ để đánh giá riêng LLM, không dùng khi nộp)")
    ap.add_argument("--asr-cache", default=None, help="jsonl transcript đã chạy sẵn (id,text,history)")
    ap.add_argument("--allow-download", action="store_true", help="cho phép tải model từ HF (chỉ khi phát triển)")
    a = ap.parse_args()
    if a.text_source == "gold":
        print("[cảnh báo] text-source=gold chỉ dùng để đo LLM trên train, KHÔNG dùng khi nộp bài.", flush=True)
    cfg = load_config(a.config)
    run(cfg, a.data, a.out, split=a.split, limit=a.limit, text_source=a.text_source,
        asr_cache=a.asr_cache, allow_download=a.allow_download)


if __name__ == "__main__":
    main()

