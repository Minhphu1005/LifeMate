# LifeMate — Hệ thống AI Agent điều khiển bằng giọng nói tiếng Việt (`full/`)

Giải pháp hoàn chỉnh (End-to-End) cho bài toán **LifeMate (Zalo AI Challenge)**: nhận âm thanh tiếng Việt 3 miền (Bắc, Trung, Nam) có pha từ địa phương/từ lóng/từ mượn tiếng Anh, tương tác nhiều bước với môi trường `zaic_tools`, và xuất file `submission.jsonl` kèm `state_hash` chuẩn xác.

---

## 1. Kiến trúc tổng thể & Ngân sách mô hình

```text
[File .wav câu hiện tại + các file .wav trong history]
                         │
                         ▼
       ┌────────────────────────────────────┐
       │  1. ASR: PhoWhisper-medium (0.77B) │  (Đã fine-tune + chuẩn hóa phiên âm/số)
       └─────────────────┬──────────────────┘
                         │  {"text": str, "history": [...]} (Unicode NFC)
                         ▼
       ┌────────────────────────────────────┐
       │  2. Prompt & Schema Engine         │  (Lọc installed_apps, rút gọn schema,
       │     + Từ điển phương ngữ 3 miền    │   bảng lịch 15 ngày, lịch sử hội thoại)
       └─────────────────┬──────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────┐
       │  3. LLM Agent: Qwen2.5-3B-Instruct │ ◄───┐  (LoRA fine-tuned trên gold +
       │     (Vòng lặp ReAct đa lượt)       │     │   noisy ASR trajectories)
       └─────────────────┬──────────────────┘     │
                         │  1 tool call / lượt    │
                         ▼                        │
       ┌────────────────────────────────────┐     │
       │  4. Kiểm duyệt env.validate()      │─────┤  (Chặn lệnh sai schema TRƯỚC KHI
       │     -> Thực thi env.call()         │─────┘   vào log -> không bị phạt U)
       └─────────────────┬──────────────────┘
                         │
                         ▼
       [submission.jsonl: {"id", "tool_calls", "state_hash"}]
```

- **Tổng tham số:** `PhoWhisper-medium` (~0.77B) + `Qwen2.5-3B-Instruct` (~3.09B) = **~3.86B** (< giới hạn 6B của đề; riêng LLM < 5B).
- **Độ dài ngữ cảnh:** Trung bình ~3.300 tokens, tối đa ~4.800 tokens (dưới xa giới hạn 16.384 tokens).
- **Tính tất định (Reproducibility):** Cố định toàn bộ seed (`Python`, `NumPy`, `PyTorch`), giải mã `greedy` (`num_beams=1`, `do_sample=False`).

---

## 2. Cấu trúc thư mục `full/`

```text
full/
├── config.yaml                  # Cấu hình duy nhất cho ASR, LLM và giới hạn hệ thống
├── run_pipeline.py              # Entrypoint chạy suy luận (Docker & Local)
├── Dockerfile                   # Đóng gói container chạy offline (--network none)
├── requirements.txt             # Thư viện phụ thuộc
├── lifemate/
│   ├── data.py                  # Tự động dò split, nạp đúng zaic_tools của từng split
│   ├── runner.py                # Điều phối pipeline, chống lỗi & timeout từng sample
│   ├── scoring.py               # Hàm tính điểm F1 + kiểm tra state_hash chuẩn theo đề
│   ├── asr/
│   │   ├── base.py              # Hợp đồng BaseASR (transcribe, transcribe_batch, transcribe_history)
│   │   ├── normalize.py         # Chuẩn hóa NFC, khôi phục từ viết tắt (VOV, VTV, HTV, SJC, SMS, RON95, Zalo), dãy số ĐT
│   │   ├── pipeline.py          # Gộp audio history + câu hiện tại thành 1 batch GPU
│   │   └── models/              # Adapter Whisper/PhoWhisper & Factory
│   └── agent/
│       ├── prompt.py            # Rút gọn schema động, từ điển phương ngữ/từ lóng, quy đổi lịch 15 ngày
│       ├── chatml.py            # Mã hóa ChatML đồng nhất giữa lúc train và lúc suy luận
│       ├── parser.py            # Bóc tách JSON tool call & ép kiểu tham số theo schema
│       ├── agent.py             # Vòng lặp ReAct có màng lọc env.validate()
│       └── llm.py               # Adapter Qwen2.5 + LoRA
├── training/
│   ├── train_all.py             # Chạy trọn gói 4 bước huấn luyện (ASR -> Cache -> SFT Data -> LLM)
│   ├── train_asr.py             # Fine-tune LoRA cho PhoWhisper-medium kèm data augmentation
│   ├── build_sft_data.py        # Tạo dữ liệu SFT đa lượt (kết hợp transcript chuẩn & transcript nhiễu ASR)
│   └── train_llm.py             # Fine-tune LoRA/QLoRA cho Qwen2.5-3B-Instruct
├── scripts/
│   ├── evaluate.py              # Chấm điểm offline trên train theo đúng công thức BTC
│   ├── oracle_check.py          # Kiểm chứng harness đạt 100.00 điểm với gold tool_calls
│   ├── run_asr_cache.py         # Xuất sẵn file transcript jsonl cho một split
│   └── download_weights.py      # Tải weights gốc vào weights/ trước khi build Docker
└── tests/
    └── test_all.py              # Bộ kiểm thử đơn vị & tích hợp toàn hệ thống
```

---

## 3. Hướng dẫn huấn luyện (trên Colab / máy có GPU)

### Cách 1: Chạy tự động trọn gói 1 lệnh
```bash
cd full
pip install -r requirements.txt

# Chạy toàn bộ: Fine-tune ASR -> Chạy ASR trên train -> Tạo SFT data -> Fine-tune LLM LoRA
python training/train_all.py --data ../train+public_test
```
*(Nếu dùng GPU Colab T4 16GB VRAM, thêm cờ `--load-4bit` để bật QLoRA tiết kiệm bộ nhớ).*

### Cách 2: Chạy từng bước riêng lẻ
```bash
# Bước 1: Fine-tune PhoWhisper-medium trên ~5.036 cặp audio-transcript của tập train
python training/train_asr.py --data ../train+public_test \
    --base vinai/PhoWhisper-medium --out weights/asr_phowhisper_ft --epochs 3

# Bước 2: Sinh transcript thực tế của tập train để LLM học cách chịu lỗi nhận dạng ASR
python scripts/run_asr_cache.py --data ../train+public_test --split train \
    --model weights/asr_phowhisper_ft --out sft_data/asr_train.jsonl --allow-download

# Bước 3: Đóng gói dữ liệu hội thoại SFT đa lượt
python training/build_sft_data.py --data ../train+public_test \
    --asr-train sft_data/asr_train.jsonl --out-dir sft_data

# Bước 4: Fine-tune LoRA cho Qwen2.5-3B-Instruct
python training/train_llm.py --data-dir sft_data \
    --base Qwen/Qwen2.5-3B-Instruct --out weights/llm_lora --epochs 2
```

---

## 4. Hướng dẫn chạy suy luận (Inference) & Đánh giá Offline

### Chạy suy luận trên `public_test` để lấy file nộp bài:
```bash
python run_pipeline.py --config config.yaml \
    --data ../train+public_test --split public_test \
    --out submission.jsonl --allow-download
```

### Đánh giá trên tập `train` (hoặc tập validation):
```bash
python run_pipeline.py --config config.yaml \
    --data ../train+public_test --split train --limit 200 \
    --out sub_train.jsonl --allow-download

python scripts/evaluate.py --data ../train+public_test \
    --split train --submission sub_train.jsonl
```

### Kiểm tra bộ test hệ thống (không cần GPU):
```bash
python tests/test_all.py --data ../train+public_test
python scripts/oracle_check.py --data ../train+public_test --limit 1545
```

---

## 5. Đóng gói Docker nộp bài (Offline 100%)

1. Đảm bảo thư mục `weights/` đã có đủ:
   - `weights/asr_phowhisper_ft` (model ASR đã gộp LoRA)
   - `weights/qwen2.5_3b` (tải bằng `python scripts/download_weights.py`)
   - `weights/llm_lora` (LoRA adapter của LLM)
2. Build & chạy thử với cờ `--network none`:
```bash
docker build -t lifemate-agent:latest .
docker run --rm --gpus all --network none \
    -v /duong/dan/public_test:/data:ro \
    -v $(pwd)/result:/result \
    lifemate-agent:latest
```
Kết quả được xuất ra tại `result/submission.jsonl` kèm báo cáo thời gian chạy ở `result/reports/latency.csv`.

