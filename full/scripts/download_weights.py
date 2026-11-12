"""Tải sẵn toàn bộ weights gốc về thư mục weights/ để đóng gói Docker chạy offline 100%.

  python scripts/download_weights.py --config config.yaml
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.runner import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from huggingface_hub import snapshot_download

    asr_cfg = cfg["asr"]
    asr_dir = Path(asr_cfg["model_path"])
    if not asr_dir.exists() or not any(asr_dir.iterdir()):
        print(f"[download] tải ASR {asr_cfg['hub_id']} -> {asr_dir}")
        snapshot_download(repo_id=asr_cfg["hub_id"], local_dir=str(asr_dir))
    else:
        print(f"[download] đã có ASR tại {asr_dir}")

    llm_cfg = cfg["llm"]
    llm_dir = Path(llm_cfg["base_model_path"])
    if not llm_dir.exists() or not any(llm_dir.iterdir()):
        print(f"[download] tải LLM {llm_cfg['hub_id']} -> {llm_dir}")
        snapshot_download(repo_id=llm_cfg["hub_id"], local_dir=str(llm_dir))
    else:
        print(f"[download] đã có LLM tại {llm_dir}")


if __name__ == "__main__":
    main()

