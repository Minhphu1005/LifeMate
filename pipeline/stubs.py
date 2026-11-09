"""Stub để Phúc chạy end-to-end khi LLM của Quân chưa xong."""
from __future__ import annotations

from typing import List


class DummyLLM:
    """Trả [] cho mọi mẫu (= out-of-scope)."""

    def generate_calls(self, user_text, history, user_context, available_tools,
                       env_caller=None, validator=None) -> List[dict]:
        return []


class ReadProbeLLM:
    """Gọi 1 read tool đầu tiên không có tham số bắt buộc qua env_caller, rồi trả list
    gồm call đó. Dùng để kiểm tra env.call / ghi log / chống trùng hoạt động."""

    def generate_calls(self, user_text, history, user_context, available_tools,
                       env_caller=None, validator=None) -> List[dict]:
        for t in available_tools:
            if t.get("kind") == "read" and not (t.get("parameters") or {}).get("required"):
                if env_caller:
                    env_caller(t["name"], {})
                return [{"name": t["name"], "arguments": {}}]
        return []

