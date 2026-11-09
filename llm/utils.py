from __future__ import annotations

from typing import List


def history_to_text(history: List[dict]) -> List[str]:
    """Làm phẳng history (list[dict]) thành list[str] các câu user, nếu cần."""
    return [t.get("text", "") for t in history if t.get("role") == "user"]

