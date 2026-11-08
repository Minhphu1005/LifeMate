# Module ASR (Phú)

Chuyển audio tiếng Việt -> text. Model hiện tại: `vinai/PhoWhisper-medium` (baseline, chưa fine-tune, WER ≈ 0.32 trên 200 mẫu train).

## Cài đặt
```
pip install -r asr/requirements.txt
```
Model tự tải từ HuggingFace lần đầu chạy (~3 GB, cần internet). Khi đóng Docker thì phải copy sẵn weights vào `/weights` và đặt `model_path` trỏ tới đó (chạy offline).

## Dùng
```python
from asr.models.asr_factory import build_asr
from asr.pipeline import transcribe_sample

asr = build_asr({"engine": "phowhisper", "model_path": "vinai/PhoWhisper-medium"})
out = transcribe_sample(asr, item, split_dir)   # item = 1 dòng của <split>.jsonl
# split_dir = thư mục chứa audio/ , ví dụ .../train hoặc .../public_test
```

Kết quả:
```python
{
  "text": "transcript câu hiện tại",
  "history": [
    {"role": "user", "text": "transcript lượt user trước"},
    {"role": "assistant", "text": "...", "tool_calls": [...], "tool_results": [...]},
  ]
}
```
- Mọi audio user (history + câu cuối) được chạy ASR trong một batch.
- Không bao giờ đọc `audio_text` (chỉ có ở train).
- Greedy, seed cố định, tất định.

## Lưu ý cho LLM
Transcript có thể sai nhẹ (tên người, từ ngắn). Nên khớp tên với kết quả tool read thay vì tin tuyệt đối vào chữ ASR.

## Đổi model
Chỉ đổi `model_path` / `engine`, không phải sửa code.

## Đánh giá WER
```
python -m asr.eval_asr --data <thư mục train+public_test> --model vinai/PhoWhisper-medium --limit 200
```

