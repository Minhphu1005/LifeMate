"""Adapter LLM (HuggingFace causal LM + LoRA đã/chưa gộp). Greedy, tất định, offline."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

from .chatml import encode_messages


def resolve_model(path: str, hub_id: Optional[str] = None, allow_download: bool = False) -> str:
    """Ưu tiên thư mục cục bộ. Khi chạy Docker (offline) phải có sẵn weights, không tải."""
    if path and Path(path).exists():
        return str(path)
    if allow_download and hub_id:
        return hub_id
    raise FileNotFoundError(
        f"Không thấy model ở '{path}'. Copy weights vào đó (Docker chạy offline) "
        f"hoặc đặt allow_hub_download: true khi phát triển.")


class HFChatLLM:
    def __init__(self, base_model: str, adapter_path: Optional[str] = None, dtype: str = "bfloat16",
                 max_new_tokens: int = 200, max_context: int = 12000, seed: int = 42,
                 device: Optional[str] = None, **_):
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        if self.device == "cpu":
            td = torch.float32
        tok_src = adapter_path if (adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists()) else base_model
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=td, attn_implementation="sdpa")
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=td)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
        self.model.config.use_cache = True
        self.model = self.model.to(self.device).eval()
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.im_end
        self.max_new_tokens = max_new_tokens
        self.max_context = max_context

    def generate(self, messages: List[dict]) -> str:
        ids, _ = encode_messages(self.tok, messages, add_generation_prompt=True)
        if len(ids) > self.max_context:
            raise ValueError(f"prompt quá dài: {len(ids)} token")
        x = self.torch.tensor([ids], device=self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                input_ids=x, attention_mask=self.torch.ones_like(x), max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, eos_token_id=self.im_end, pad_token_id=self.pad)
        return self.tok.decode(out[0, x.shape[1]:], skip_special_tokens=True).strip()

