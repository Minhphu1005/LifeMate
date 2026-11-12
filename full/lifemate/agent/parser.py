"""Phân tích đầu ra LLM -> tool call, và làm sạch tham số theo schema (không thêm dữ kiện mới)."""
from __future__ import annotations

import json
from typing import Optional, Tuple

from .prompt import END


def parse_call(text: str) -> Optional[dict]:
    """Trả về {"name", "arguments"} hoặc None nếu mô hình báo xong / không có lệnh hợp lệ."""
    t = (text or "").strip()
    if not t or t.startswith(END):
        return None
    i = t.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"].strip(), "arguments": args}


def _coerce(value, spec: dict):
    t = spec.get("type")
    try:
        if t == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                s = value.replace(".", "").replace(",", "").strip()
                if s.lstrip("-").isdigit():
                    return int(s)
        elif t == "number":
            if isinstance(value, str):
                return float(value.replace(",", ".").strip())
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        elif t == "boolean":
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        elif t == "string":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
        elif t == "array":
            if not isinstance(value, list):
                return [value] if isinstance(value, (str, int, float)) else value
    except Exception:
        pass
    return value


def clean_arguments(args: dict, tool_spec: Optional[dict]) -> dict:
    """Bỏ giá trị null, bỏ khóa không có trong schema, ép kiểu theo schema. Không bịa giá trị mới."""
    if not tool_spec:
        return args
    props = tool_spec.get("parameters", {}).get("properties", {})
    out = {}
    for k, v in args.items():
        if v is None or k not in props:
            continue
        out[k] = _coerce(v, props[k])
    return out

