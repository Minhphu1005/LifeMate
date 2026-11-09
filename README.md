# Pipeline & đánh giá (Phúc)

Cấu trúc dự án:
```
asr/            Phú   – ASR (build_asr, transcribe_sample)
llm/            Quân  – LLM (build_llm, BaseLLMAgent.generate_calls); hiện có engine "dummy"
pipeline/       Phúc  – catalog.py (lọc tool), runner.py (vòng lặp), stubs.py
run_pipeline.py Phúc  – entrypoint Docker
evaluate.py     Phúc  – chấm F1 offline
config.yaml     đổi engine/model_path là đủ để ghép
tests/          test offline bằng dataset + zaic_tools giả
```

## Chạy thử (không cần model)
```
python -m unittest tests.test_pipeline_mock -v
python run_pipeline.py --config config.yaml --data-dir <ROOT> --split public_test --out out/submission.jsonl --limit 20
python evaluate.py --self-test --data-dir <ROOT> --split train      # nộp = gold -> 100 điểm
python evaluate.py --pred out/submission.jsonl --data-dir <ROOT> --split train
```
`<ROOT>` = thư mục CHA chứa `<split>/` và `<split>_environment/` (Colab: `train+public_test`).

## Khi ASR/LLM xong
Chỉ sửa `config.yaml`: `asr.engine/model_path`, `llm.engine/model_path`. Quân thêm
`llm/models/hf_llm.py` (class `HFLLMAgent`) là `llm_factory` tự nạp được.

## Docker
```
docker build -t lifemate .
docker run --gpus all --network none -v <ROOT>:/data -v <out>:/result lifemate
```
Cần thư mục `weights/` ở gốc repo (không commit lên git) trước khi build.

