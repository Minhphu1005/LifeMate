"""Phân tích đầu ra LLM -> tool call, và làm sạch tham số theo schema (không thêm dữ kiện mới)."""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from .prompt import END

_PHONE_RE = re.compile(r"\b0\d{8,11}\b")


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
                value = [value] if isinstance(value, (str, int, float)) else value
            item_spec = spec.get("items") or {}
            if isinstance(value, list) and item_spec.get("type") == "object":
                ip = item_spec.get("properties") or {}
                cleaned_list = []
                for elem in value:
                    if isinstance(elem, dict):
                        cleaned_list.append({
                            ek: (_coerce(ev, ip[ek]) if ek in ip else ev)
                            for ek, ev in elem.items() if ev is not None
                        })
                    else:
                        cleaned_list.append(elem)
                return cleaned_list
    except Exception:
        pass
    return value


def _str_edit_dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def clean_arguments(args: dict, tool_spec: Optional[dict], user_text: str = "") -> dict:
    """Bỏ giá trị null, bỏ khóa không có trong schema, ép kiểu theo schema và khớp lại chuỗi SĐT nếu LLM nuốt 1 chữ số."""
    if not tool_spec:
        return args
    props = tool_spec.get("parameters", {}).get("properties", {})
    text_phones = _PHONE_RE.findall(user_text or "")
    out = {}
    for k, v in args.items():
        if v is None or k not in props:
            continue
        cv = _coerce(v, props[k])
        if k in ("phone", "phone_number") and isinstance(cv, str) and text_phones and cv not in text_phones:
            # Nếu LLM tự cắt 1 chữ số (ví dụ SĐT 11 số trong ZAIC bị cắt còn 10 số) -> trả về đúng số người dùng nói
            best = min(text_phones, key=lambda p: _str_edit_dist(cv, p))
            if _str_edit_dist(cv, best) <= 2:
                cv = best
        out[k] = cv
    return out

