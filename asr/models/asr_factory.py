"""Factory: đọc config.yaml -> nạp đúng adapter ASR (cắm-rút model)."""
from __future__ import annotations

from ..base import BaseASR


class DummyASR(BaseASR):
    """Trả về chuỗi rỗng; dùng để test pipeline không cần model."""
    def transcribe(self, audio_path, metadata=None):
        return ""


def build_asr(cfg: dict) -> BaseASR:
    """cfg = phần `asr:` trong config.yaml, ví dụ
    {engine: whisper, model_path: /weights/whisper, language: vi, ...}"""
    engine = cfg.get("engine", "whisper")
    if engine == "dummy":
        return DummyASR()
    if engine in ("whisper", "phowhisper"):  # cả hai đều là kiến trúc Whisper (HF)
        from .whisper_hf import WhisperASR
        return WhisperASR(**cfg)
    raise ValueError(f"ASR engine không hỗ trợ: {engine}")

