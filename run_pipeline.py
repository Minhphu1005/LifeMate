"""Điểm vào duy nhất (Docker entrypoint): python run_pipeline.py --config config.yaml"""
import argparse

import yaml

from pipeline.runner import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", help="ghi đè paths.data_dir (thư mục CHA chứa <split>/)")
    ap.add_argument("--split", help="ghi đè paths.split")
    ap.add_argument("--out", help="ghi đè paths.out")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N mẫu đầu (dev)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    p = cfg.setdefault("paths", {})
    data_dir = a.data_dir or p.get("data_dir", "/data")
    split = a.split or p.get("split", "private_test")
    out = a.out or p.get("out", "/result/submission.jsonl")
    run(cfg, split, data_dir, out, a.limit)


if __name__ == "__main__":
    main()

