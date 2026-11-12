FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DATA_DIR=/data \
    OUT_FILE=/result/submission.jsonl

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY lifemate /workspace/lifemate
COPY scripts /workspace/scripts
COPY training /workspace/training
COPY weights /workspace/weights
COPY config.yaml /workspace/config.yaml
COPY run_pipeline.py /workspace/run_pipeline.py

CMD ["python", "/workspace/run_pipeline.py", "--config", "/workspace/config.yaml"]

