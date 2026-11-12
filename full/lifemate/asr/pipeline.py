"""Điểm vào cho pipeline ASR:
- Gộp audio history + câu hiện tại thành 1 batch GPU.
- Lọc bỏ các lượt audio history thuần xã giao (khi lượt assistant kế tiếp không gọi tool và không đặt câu hỏi '?').
- Tự động trích từ khóa ngữ cảnh (tên người dùng, địa chỉ, tên riêng trong history) để hỗ trợ ASR.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BaseASR
from .normalize import clean


def _is_chitchat_pair(user_turn: dict, next_turn: dict | None) -> bool:
    """Lượt user trong history là xã giao nếu lượt assistant ngay sau đó không gọi tool và không hỏi lại ('?')."""
    if not next_turn or next_turn.get("role") != "assistant":
        return False
    if next_turn.get("tool_calls"):
        return False
    asst_text = next_turn.get("text") or ""
    if "?" in asst_text:
        return False
    return True


def extract_context_keywords(item: dict, max_chars: int = 160) -> str:
    """Trích từ khóa hợp lệ từ context.user_profile để làm gợi ý từ vựng (hotwords) cho ASR."""
    ctx = item.get("context") or {}
    prof = ctx.get("user_profile") or {}
    bits = []
    for k in ("name", "home_address", "work_address", "city"):
        v = prof.get(k)
        if v and isinstance(v, str):
            bits.append(v)
    joined = ", ".join(bits)
    return joined[:max_chars]


def transcribe_sample(asr: BaseASR, item: dict, split_dir,
                      skip_chitchat_history: bool = True,
                      use_context_prompt: bool = False) -> dict:
    """Gộp audio history cần thiết + câu hiện tại thành MỘT lần batch.

    Trả về:
      {
        "text": str,                       # transcript câu hiện tại
        "history": [                       # giữ nguyên thứ tự turn gốc
          {"role": "user", "text": str},
          {"role": "assistant", "text": str, "tool_calls": [...], "tool_results": [...]},
        ]
      }
    Không bao giờ đọc "audio_text" (leak từ train).
    """
    root = Path(split_dir)
    history = item.get("history") or []
    user_idx = []
    for i, t in enumerate(history):
        if t.get("role") == "user" and t.get("audio"):
            nxt = history[i + 1] if i + 1 < len(history) else None
            if skip_chitchat_history and _is_chitchat_pair(t, nxt):
                continue
            user_idx.append(i)

    paths: List[str] = [str(root / history[i]["audio"]) for i in user_idx]
    paths.append(str(root / item["audio"]))

    meta = {"prompt": extract_context_keywords(item)} if use_context_prompt else None
    try:
        texts = asr.transcribe_batch(paths, metadata=meta)
    except TypeError:
        texts = asr.transcribe_batch(paths)
    hist_text = dict(zip(user_idx, texts[:-1]))

    out_hist = []
    for i, t in enumerate(history):
        if t.get("role") == "user":
            out_hist.append({"role": "user", "text": hist_text.get(i, "")})
        else:
            out_hist.append({"role": t.get("role", "assistant"),
                             "text": clean(t.get("text", "")),
                             "tool_calls": t.get("tool_calls", []),
                             "tool_results": t.get("tool_results", [])})
    return {"text": texts[-1], "history": out_hist}
