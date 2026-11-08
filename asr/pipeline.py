"""Điểm vào cho pipeline: 1 sample -> transcript câu hiện tại + history đầy đủ.

    from asr.models.asr_factory import build_asr
    from asr.pipeline import transcribe_sample
    asr = build_asr(cfg["asr"])
    out = transcribe_sample(asr, item, split_dir)   # split_dir = thư mục chứa audio/
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import BaseASR
from .normalize import clean


def transcribe_sample(asr: BaseASR, item: dict, split_dir) -> dict:
    """Gộp audio history + câu hiện tại thành MỘT lần batch (nhanh hơn chạy lẻ).

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
    user_idx = [i for i, t in enumerate(history) if t.get("role") == "user" and t.get("audio")]
    paths: List[str] = [str(root / history[i]["audio"]) for i in user_idx]
    paths.append(str(root / item["audio"]))

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

