"""Hợp đồng module ASR (theo zalo-ý-tưởng-final.txt, mục III.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


class BaseASR:
    def transcribe(self, audio_path: str, metadata: Optional[dict] = None) -> str:
        """Trả về transcript tiếng Việt: Unicode NFC, đã strip."""
        raise NotImplementedError

    def transcribe_batch(self, audio_paths: Iterable[str],
                         metadata: Optional[dict] = None) -> List[str]:
        """Mặc định gọi tuần tự; adapter có thể ghi đè để chạy batch (nhanh hơn)."""
        return [self.transcribe(p, metadata) for p in audio_paths]

    def transcribe_history(self, history: List[dict],
                           metadata: Optional[dict] = None) -> List[str]:
        """Transcript của các turn `user` trong history (theo thứ tự).

        - Chỉ chạy ASR trên turn có "audio"; turn assistant bị bỏ qua ở hàm này
          (xem asr.pipeline.transcribe_sample để lấy history đầy đủ cho LLM).
        - LUÔN bỏ qua "audio_text" (chỉ có ở train, không có ở test).
        - Đường dẫn audio tương đối được ghép với metadata["audio_root"].
        """
        root = Path((metadata or {}).get("audio_root", "."))
        paths = [str(root / t["audio"]) for t in history
                 if t.get("role") == "user" and t.get("audio")]
        return self.transcribe_batch(paths, metadata) if paths else []
