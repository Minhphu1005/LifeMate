"""Fine-tune LoRA cho LLM agent trên sft_data (xem build_sft_data.py). Chạy trên Colab/GPU.

  pip install -q transformers accelerate peft datasets sentencepiece
  python training/train_llm.py --data-dir sft_data --base Qwen/Qwen2.5-3B-Instruct --out weights/llm_lora --epochs 2

GPU: A100/L4 (bf16) chạy tốt 3B; T4 dùng --base Qwen/Qwen2.5-1.5B-Instruct hoặc thêm --load-4bit.
Kết quả: thư mục adapter (LoRA) + tokenizer. Dùng bằng llm.adapter_path trong config.yaml.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.agent.chatml import encode_messages  # noqa: E402


class SeqDS:
    def __init__(self, path, tok, max_len):
        self.rows = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids, labels = encode_messages(tok, json.loads(line)["messages"])
                if len(ids) > max_len:
                    skipped += 1
                    continue
                self.rows.append({"input_ids": ids, "labels": labels})
        print(f"[data] {path}: {len(self.rows)} chuỗi, bỏ {skipped} chuỗi quá dài (> {max_len})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collator(pad_id):
    import torch

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="sft_data")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="weights/llm_lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-acc", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (tiết kiệm VRAM, hơi chậm)")
    ap.add_argument("--max-train", type=int, default=0, help="thử nhanh: chỉ dùng N chuỗi")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import numpy as np
    import torch
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    kw = dict(torch_dtype=dtype)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        kw["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
    else:
        kw["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train = SeqDS(os.path.join(a.data_dir, "sft_train.jsonl"), tok, a.max_len)
    val = SeqDS(os.path.join(a.data_dir, "sft_val.jsonl"), tok, a.max_len)
    if a.max_train:
        train.rows = train.rows[: a.max_train]
        val.rows = val.rows[: max(4, a.max_train // 8)]

    eval_kw = {"eval_strategy": "epoch"} if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames else {"evaluation_strategy": "epoch"}
    args = TrainingArguments(
        output_dir=os.path.join(a.out, "ckpt"), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_acc, learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", warmup_steps=10, weight_decay=0.0, bf16=bf16, fp16=not bf16,
        logging_steps=5, save_strategy="no", report_to=[], seed=a.seed,
        remove_unused_columns=False, dataloader_num_workers=0, optim="adamw_torch", **eval_kw)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=make_collator(tok.pad_token_id if tok.pad_token_id is not None else 0))
    t0 = time.time()
    trainer.train()
    m = trainer.evaluate()
    print(f"[done] {(time.time() - t0) / 60:.1f} phút | val loss={m.get('eval_loss'):.4f} "
          f"ppl={math.exp(m.get('eval_loss', 0)):.3f}")
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("đã lưu LoRA adapter vào", a.out)


if __name__ == "__main__":
    main()

