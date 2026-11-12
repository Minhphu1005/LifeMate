"""Chạy trọn gói toàn bộ quy trình huấn luyện & đánh giá của dự án LifeMate:
  Bước 1: Fine-tune PhoWhisper-medium trên tập train (giọng 3 miền + augmentation) -> weights/asr_phowhisper_ft
  Bước 2: Chạy ASR trên tập train -> sft_data/asr_train.jsonl (để LLM học cả trên transcript có nhiễu ASR)
  Bước 3: Tạo dữ liệu SFT đa lượt (gold + noisy ASR + từ điển phương ngữ + schema rút gọn) -> sft_data/
  Bước 4: Fine-tune LoRA cho Qwen2.5-3B-Instruct -> weights/llm_lora

Ví dụ chạy đầy đủ trên Colab / máy có GPU:
  python training/train_all.py --data ../train+public_test

Chạy thử nhanh (smoke test ~3 phút) để kiểm tra toàn bộ luồng:
  python training/train_all.py --data ../train+public_test --smoke
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd):
    print("\n>>>", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Thư mục chứa train/ và public_test/")
    ap.add_argument("--asr-base", default="vinai/PhoWhisper-medium")
    ap.add_argument("--llm-base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--asr-out", default="weights/asr_phowhisper_ft")
    ap.add_argument("--llm-out", default="weights/llm_lora")
    ap.add_argument("--sft-dir", default="sft_data")
    ap.add_argument("--skip-asr-ft", action="store_true", help="Bỏ qua bước fine-tune ASR")
    ap.add_argument("--skip-asr-cache", action="store_true", help="Bỏ qua bước sinh transcript ASR cho train")
    ap.add_argument("--load-4bit", action="store_true", help="Dùng QLoRA 4-bit cho LLM (khi chạy trên GPU T4 16GB)")
    ap.add_argument("--smoke", action="store_true", help="Chạy thử nhanh trên vài chục mẫu")
    a = ap.parse_args()

    py = sys.executable
    sft_dir = Path(a.sft_dir)
    sft_dir.mkdir(parents=True, exist_ok=True)
    asr_cache = sft_dir / "asr_train.jsonl"

    # 1. Fine-tune ASR
    if not a.skip_asr_ft:
        cmd = [py, "training/train_asr.py", "--data", a.data, "--base", a.asr_base, "--out", a.asr_out]
        cmd += ["--epochs", "1", "--max-train", "32"] if a.smoke else ["--epochs", "3"]
        sh(cmd)

    # 2. Chạy ASR trên train để lấy transcript thực tế đưa vào huấn luyện LLM
    if not a.skip_asr_cache:
        cmd = [py, "scripts/run_asr_cache.py", "--data", a.data, "--split", "train",
               "--out", str(asr_cache), "--allow-download"]
        if Path(a.asr_out).exists():
            cmd += ["--model", a.asr_out]
        if a.smoke:
            cmd += ["--limit", "40"]
        sh(cmd)

    # 3. Dựng dữ liệu SFT (gold + noisy ASR)
    cmd = [py, "training/build_sft_data.py", "--data", a.data, "--out-dir", str(sft_dir),
           "--tokenizer", a.llm_base]
    if asr_cache.exists():
        cmd += ["--asr-train", str(asr_cache)]
    sh(cmd)

    # 4. Fine-tune LoRA cho LLM Agent
    cmd = [py, "training/train_llm.py", "--data-dir", str(sft_dir), "--base", a.llm_base, "--out", a.llm_out]
    if a.load_4bit:
        cmd.append("--load-4bit")
    cmd += ["--epochs", "1", "--max-train", "24"] if a.smoke else ["--epochs", "2"]
    sh(cmd)

    print("\n[HOÀN TẤT] Đã huấn luyện xong cả ASR và LLM Agent!")
    print(f"  - ASR weights : {a.asr_out}")
    print(f"  - LLM adapter : {a.llm_out}")


if __name__ == "__main__":
    main()

