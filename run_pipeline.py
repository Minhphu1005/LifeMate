import argparse
from pathlib import Path

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

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg.setdefault("paths", {})
    data_dir = Path(a.data_dir or p.get("data_dir", "/data")).expanduser()
    split = a.split or p.get("split", "private_test")
    if not a.split and not (data_dir / split / f"{split}.jsonl").exists():
        for cand in ("public_test", "train"):
            if (data_dir / cand / f"{cand}.jsonl").exists() or (data_dir / data_dir.name / cand / f"{cand}.jsonl").exists():
                split = cand
                break

    out = a.out or p.get("out", "/result/submission.jsonl")
    if not a.out and str(out).startswith("/result/") and not Path("/result").exists():
        out = "result/submission.jsonl"
        if str(p.get("reports", "")).startswith("/result/"):
            p["reports"] = "result/reports"

    run(cfg, split, data_dir, out, a.limit)


if __name__ == "__main__":
    main()

