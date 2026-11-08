"""Fine-tune PhoWhisper bằng LoRA trên train (audio chính + audio history).

Chạy trên Colab (GPU T4 trở lên):
  !pip install -q transformers accelerate peft soundfile librosa
  %cd /content/drive/MyDrive
  !python -m asr.finetune --data /content/dataset/train+public_test \
      --base vinai/PhoWhisper-medium --out /content/drive/MyDrive/pho_ft --epochs 3

Chỉ dùng train. Tách validation 15% theo sample id (cố định seed 42).
Cuối cùng: gộp LoRA vào model, lưu ở --out, in WER trên validation (trước và sau).
Dùng model mới: build_asr({"engine": "phowhisper", "model_path": "<out>"}).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .eval_asr import _edit, _norm
from .models.whisper_hf import _load_audio


def build_pairs(sdir: Path, val_frac: float, seed: int):
    rows = [json.loads(l) for l in open(sdir / "train.jsonl", encoding="utf-8")]
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[: int(len(ids) * val_frac)])
    train, val = [], []
    for r in rows:
        pairs = [(str(sdir / r["audio"]), r["audio_text"])]
        pairs += [(str(sdir / h["audio"]), h["audio_text"])
                  for h in r["history"] if h.get("role") == "user" and h.get("audio")]
        (val if r["id"] in val_ids else train).extend(pairs)
    return train, val


class DS:
    def __init__(self, pairs, proc):
        self.pairs, self.proc = pairs, proc

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, text = self.pairs[i]
        feats = self.proc.feature_extractor(_load_audio(path), sampling_rate=16000).input_features[0]
        labels = self.proc.tokenizer(text).input_ids
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    proc: object
    start_token_id: int

    def __call__(self, batch):
        import torch
        feats = torch.tensor([b["input_features"] for b in batch])
        lab = self.proc.tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.start_token_id).all():  # decoder tự thêm token bắt đầu
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def wer(model, proc, pairs, device, bs=8, limit=300):
    import torch
    pairs = pairs[:limit]
    model.eval()
    we = wn = 0
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        f = proc.feature_extractor([_load_audio(p) for p, _ in chunk], sampling_rate=16000,
                                   return_tensors="pt").input_features.to(device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(f, language="vi", task="transcribe", do_sample=False,
                                 num_beams=1, max_new_tokens=128, no_repeat_ngram_size=4)
        for (_, ref), hyp in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
            r, h = _norm(ref).split(), _norm(hyp).split()
            we += _edit(r, h); wn += len(r)
    return we / max(wn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="vinai/PhoWhisper-medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-4, help="LoRA; dùng ~1e-5 nếu --full")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--full", action="store_true", help="fine-tune toàn bộ (cần A100)")
    ap.add_argument("--max-train", type=int, default=0, help="giới hạn số cặp train (để thử nhanh)")
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    root = Path(a.data)
    if not (root / "train").exists() and (root / root.name / "train").exists():
        root = root / root.name
    train, val = build_pairs(root / "train", a.val_frac, 42)
    if a.max_train:
        train = train[: a.max_train]
    print(f"train pairs {len(train)} | val pairs {len(val)}")

    device = "cuda"
    proc = WhisperProcessor.from_pretrained(a.base, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base).to(device)
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False

    print("WER val TRƯỚC fine-tune:", round(wer(model, proc, val, device), 4))
    model.config.use_cache = False

    if not a.full:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
        model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(a.out) / "ckpt"), per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_acc, learning_rate=a.lr, warmup_steps=50,
        num_train_epochs=a.epochs, fp16=True, gradient_checkpointing=True,
        logging_steps=20, save_strategy="no", report_to=[], remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=2, seed=42,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train, proc),
                             data_collator=Collator(proc, model.config.decoder_start_token_id))
    t0 = time.time()
    trainer.train()
    print(f"train xong sau {(time.time() - t0) / 60:.1f} phút")

    if not a.full:
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.half()
    print("WER val SAU fine-tune:", round(wer(model, proc, val, device), 4))
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)
    json.dump([p for p, _ in val], open(Path(a.out) / "val_audio.json", "w"), ensure_ascii=False)
    print("đã lưu model vào", a.out)


if __name__ == "__main__":
    main()

