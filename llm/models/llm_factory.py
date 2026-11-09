"""Factory: đọc config.yaml (phần `llm:`) -> nạp đúng adapter LLM (cắm-rút model)."""
from __future__ import annotations

from ..base import BaseLLMAgent


class DummyLLMAgent(BaseLLMAgent):
    """Luôn trả [] (coi như out-of-scope); dùng để test pipeline khi LLM chưa xong."""

    def generate_calls(self, user_text, history, user_context, available_tools,
                       env_caller=None, validator=None):
        return []


def build_llm(cfg: dict) -> BaseLLMAgent:
    """cfg = {engine: dummy | hf, model_path: /weights/llm, ...}"""
    engine = cfg.get("engine", "dummy")
    if engine == "dummy":
        return DummyLLMAgent()
    if engine == "hf":
        # Quân: tạo llm/models/hf_llm.py với class HFLLMAgent(BaseLLMAgent)
        from .hf_llm import HFLLMAgent
        return HFLLMAgent(**cfg)
    raise ValueError(f"LLM engine không hỗ trợ: {engine}")

