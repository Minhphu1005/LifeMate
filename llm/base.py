"""Hợp đồng module LLM (Quân). Xem cong-viec-2-LLM-input-output.txt."""
from __future__ import annotations

from typing import Callable, List, Optional


class BaseLLMAgent:
    def generate_calls(self,
                       user_text: str,
                       history: List[dict],
                       user_context: dict,
                       available_tools: List[dict],
                       env_caller: Optional[Callable] = None,
                       validator: Optional[Callable] = None) -> List[dict]:
        """Trả về list[{"name": str, "arguments": dict}] theo thứ tự thực thi.

        - history: out["history"] của asr.pipeline.transcribe_sample (list[dict]).
        - env_caller(name, args): mặc định cho read tool; nếu config bật
          allow_write_in_env_caller thì cho cả write đã pass validator (cấm retry).
        - validator(name, args): dry-run kiểm tra schema, không ghi log.
        - Nếu thiếu tham số bắt buộc non-ref -> trả ngay [sys.ask_missing_info],
          không gọi read trước.
        - Không được raise; lỗi -> trả [] hoặc kết quả tốt nhất đã có.
        """
        raise NotImplementedError

