"""Định dạng ChatML (Qwen) với mặt nạ huấn luyện. Dùng chung cho train và inference.

Mỗi message được token hóa RIÊNG rồi nối lại, nên chuỗi token lúc suy luận giống hệt lúc huấn luyện.
Chỉ phần nội dung assistant (kèm <|im_end|>) được tính loss.
"""
from __future__ import annotations

from typing import List, Tuple

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def encode_messages(tokenizer, messages: List[dict], add_generation_prompt: bool = False,
                    train_on_last_only: bool = False) -> Tuple[List[int], List[int]]:
    """Trả về (input_ids, labels). labels = -100 ở phần không huấn luyện.

    - add_generation_prompt=True: kết thúc bằng '<|im_start|>assistant\\n' (dùng khi sinh).
    - train_on_last_only: chỉ tính loss ở message assistant cuối.
    """
    ids: List[int] = []
    labels: List[int] = []
    last_asst = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    for i, m in enumerate(messages):
        head = enc(f"{IM_START}{m['role']}\n")
        ids += head
        labels += [-100] * len(head)
        body = enc(m["content"] + IM_END)
        tail = enc("\n")
        learn = m["role"] == "assistant" and (not train_on_last_only or i == last_asst)
        ids += body
        labels += body if learn else [-100] * len(body)
        ids += tail
        labels += [-100] * len(tail)
    if add_generation_prompt:
        head = enc(f"{IM_START}assistant\n")
        ids += head
        labels += [-100] * len(head)
    return ids, labels

