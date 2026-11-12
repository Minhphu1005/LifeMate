"""Fine-tune PhoWhisper (LoRA) trên train + augmentation giọng (tốc độ, âm lượng, nhiễu) để bền với giọng 3 miền.

  pip install -q transformers accelerate peft soundfile librosa
  python training/train_asr.py --data ../train+public_test --base vinai/PhoWhisper-medium \
      --out weights/asr_phowhisper_ft --epochs 3

- Dùng audio chính + audio user trong history của train (≈5.000 cặp). KHÔNG dùng public/private_test.
- Mặc định tách 12% mẫu làm validation (theo sample id); --val-frac 0 để huấn luyện bản cuối trên toàn bộ.
- Kết quả: model đã gộp LoRA, dùng ngay bằng asr.model_path trong config.yaml.
"""
import argparse
import json
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.asr.models.whisper_hf import _load_audio  # noqa: E402
from lifemate.data import locate_split, read_jsonl  # noqa: E402


def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    return " ".join("".join(c if (c.isalnum() or c.isspace()) else " " for c in s).split())


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def build_pairs(sdir, val_frac, seed):
    rows = list(read_jsonl(sdir / "train.jsonl"))
    ids = sorted(r["id"] for r in rows)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[: int(len(ids) * val_frac)])
    tr, va = [], []
    for r in rows:
        pairs = [(str(sdir / r["audio"]), r["audio_text"])]
        pairs += [(str(sdir / h["audio"]), h["audio_text"]) for h in r["history"]
                  if h.get("role") == "user" and h.get("audio")]
        (va if r["id"] in val_ids else tr).extend(pairs)
    return tr, va


def augment(wav):
    import numpy as np
    if random.random() < 0.6:                       # đổi tốc độ nói 0.9–1.1
        f = random.uniform(0.9, 1.1)
        n = int(len(wav) / f)
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype("float32")
    if random.random() < 0.5:                       # đổi âm lượng
        wav = wav * random.uniform(0.6, 1.4)
    if random.random() < 0.5:                       # nhiễu nền nhẹ, SNR 15–35 dB
        snr = random.uniform(15, 35)
        p = float((wav ** 2).mean()) + 1e-9
        wav = wav + np.random.randn(len(wav)).astype("float32") * (p / (10 ** (snr / 10))) ** 0.5
    return np.clip(wav, -1.0, 1.0).astype("float32")


class DS:
    def __init__(self, pairs, proc, aug):
        self.pairs, self.proc, self.aug = pairs, proc, aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, text = self.pairs[i]
        wav = _load_audio(path)
        if self.aug:
            wav = augment(wav)
        feats = self.proc.feature_extractor(wav, sampling_rate=16000).input_features[0]
        return {"input_features": feats, "labels": self.proc.tokenizer(text).input_ids}


@dataclass
class Collator:
    proc: object
    start_id: int

    def __call__(self, batch):
        import torch
        feats = torch.tensor([b["input_features"] for b in batch])
        lab = self.proc.tokenizer.pad({"input_ids": [b["labels"] for b in batch]}, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.start_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def eval_wer(model, proc, pairs, device, bs=8, limit=300):
    import torch
    pairs = pairs[:limit]
    if not pairs:
        return float("nan")
    model.eval()
    we = wn = 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        f = proc.feature_extractor([_load_audio(p) for p, _ in ch], sampling_rate=16000,
                                   return_tensors="pt").input_features.to(device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(f, language="vi", task="transcribe", do_sample=False, num_beams=1,
                                 max_new_tokens=128, no_repeat_ngram_size=4)
        for (_, ref), hyp in zip(ch, proc.batch_decode(ids, skip_special_tokens=True)):
            r, h = norm(ref).split(), norm(hyp).split()
            we += edit(r, h); wn += len(r)
    return we / max(wn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="vinai/PhoWhisper-medium")
    ap.add_argument("--out", default="weights/asr_phowhisper_ft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-acc", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    split, sdir, _ = locate_split(a.data, "train")
    train, val = build_pairs(sdir, a.val_frac, 42)
    if a.max_train:
        train = train[: a.max_train]
    print(f"train pairs {len(train)} | val pairs {len(val)}")

    dev = "cuda"
    proc = WhisperProcessor.from_pretrained(a.base, language="vi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.base).to(dev)
    model.generation_config.language, model.generation_config.task = "vi", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    if val:
        print("WER val TRƯỚC:", round(eval_wer(model, proc, val, dev), 4))
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(a.out, "ckpt"), per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_acc, learning_rate=a.lr, warmup_steps=50, num_train_epochs=a.epochs,
        fp16=True, gradient_checkpointing=True, logging_steps=20, save_strategy="no", report_to=[],
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=2, seed=42,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=DS(train, proc, not a.no_aug),
                        data_collator=Collator(proc, model.config.decoder_start_token_id))
    t0 = time.time()
    tr.train()
    print(f"train xong sau {(time.time() - t0) / 60:.1f} phút")
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.half()
    if val:
        print("WER val SAU:", round(eval_wer(model, proc, val, dev), 4))
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)
    print("đã lưu ASR vào", a.out)


if __name__ == "__main__":
    main()

