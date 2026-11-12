"""Kiểm tra khóa ngoại (x-ref) & Tự động ánh xạ Tên -> ID từ kết quả Read Tool.

Tại sao cần module này?
Trong zaic_tools, env.validate() chỉ kiểm tra kiểu dữ liệu (str/int), KHÔNG kiểm tra ID có tồn tại không.
Nếu LLM truyền nhầm tên hiển thị ('đèn phòng khách') vào trường device_id hoặc đoán mò ID khi chưa gọi
read tool, env.call() vẫn chạy nhưng trả về done=False, làm tăng U và lệch state_hash (0 điểm).

Module này:
1. Thu thập toàn bộ ID và bản ghi {id, name/title/label...} từ history.tool_results và các bước read đã chạy.
2. Nếu LLM điền tên hiển thị vào trường x-ref -> tự động đổi sang đúng mã ID.
3. Nếu LLM đoán mò ID khi chưa có kết quả read -> chặn trước khi vào env.call() và nhắc LLM gọi read tool.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Set, Tuple


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).lower().split())


def _walk_results(obj, seen_ids: Set[str], records: List[Tuple[str, dict]], bucket: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (k in ("id", "merchant_id") or k.endswith("_id")) and isinstance(v, str) and v:
                seen_ids.add(v)
        if isinstance(obj.get("id"), str):
            records.append((bucket, obj))
        for k, v in obj.items():
            _walk_results(v, seen_ids, records, bucket=k if isinstance(v, list) else bucket)
    elif isinstance(obj, list):
        for x in obj:
            _walk_results(x, seen_ids, records, bucket=bucket)


def collect_state_knowledge(history: Optional[List[dict]], steps: List[dict]) -> Tuple[Set[str], List[Tuple[str, dict]], Set[str]]:
    """Trả về (seen_ids, records, called_apps)."""
    seen_ids: Set[str] = set()
    records: List[Tuple[str, dict]] = []
    called_apps: Set[str] = set()

    for h in history or []:
        for c in h.get("tool_calls") or []:
            if isinstance(c.get("name"), str) and "." in c["name"]:
                called_apps.add(c["name"].split(".", 1)[0])
            for k, v in (c.get("arguments") or {}).items():
                if isinstance(v, str) and v:
                    seen_ids.add(v)
        _walk_results(h.get("tool_results") or [], seen_ids, records)

    for s in steps:
        if "error" in s:
            continue
        cname = s["call"]["name"]
        if "." in cname:
            called_apps.add(cname.split(".", 1)[0])
        _walk_results(s.get("result"), seen_ids, records)

    return seen_ids, records, called_apps


def _match_record_id(val: str, ref_table: str, id_key: str, records: List[Tuple[str, dict]]) -> Optional[str]:
    """Tìm mã ID nếu LLM truyền nhầm tên hiển thị (name/title/label/provider) thay vì ID."""
    target = _norm(val)
    if not target:
        return None
    table_key = ref_table.split(".")[-1] if "." in ref_table else ref_table
    # Ưu tiên bản ghi nằm đúng bảng x-ref (ví dụ 'devices', 'contacts', 'beneficiaries')
    for prefer_bucket in (True, False):
        exact_hits = []
        sub_hits = []
        for bucket, rec in records:
            if prefer_bucket and table_key and bucket and bucket != table_key:
                continue
            rid = rec.get(id_key)
            if not isinstance(rid, str) or not rid:
                continue
            for field in ("name", "title", "label", "provider", "counterparty", "room", "phone", "phone_number"):
                fv = rec.get(field)
                if isinstance(fv, str) and fv:
                    nf = _norm(fv)
                    if nf == target:
                        exact_hits.append(rid)
                    elif len(target) >= 3 and (target in nf or nf in target):
                        sub_hits.append(rid)
        uniq_exact = list(dict.fromkeys(exact_hits))
        if len(uniq_exact) == 1:
            return uniq_exact[0]
        uniq_sub = list(dict.fromkeys(sub_hits))
        if len(uniq_sub) == 1:
            return uniq_sub[0]
    return None


def verify_and_repair_xrefs(call: dict, tool_spec: Optional[dict],
                            history: Optional[List[dict]], steps: List[dict]) -> Optional[str]:
    """Kiểm tra và tự động sửa các tham số x-ref trong `call['arguments']`.

    - Trả về None nếu hợp lệ (hoặc đã tự sửa Tên -> ID thành công).
    - Trả về chuỗi thông báo lỗi nếu LLM đoán mò ID khi chưa gọi read tool.
    """
    if not tool_spec:
        return None
    props = tool_spec.get("parameters", {}).get("properties", {})
    args = call.get("arguments") or {}
    xref_params = [(p, s) for p, s in props.items() if "x-ref" in s and p in args]
    if not xref_params:
        return None

    seen_ids, records, called_apps = collect_state_knowledge(history, steps)
    for param, spec in xref_params:
        val = args.get(param)
        if not isinstance(val, str) or not val:
            continue
        if val in seen_ids:
            continue
        desc = spec.get("description") or ""
        id_key = "merchant_id" if "merchant_id" in desc else "id"
        mapped = _match_record_id(val, spec["x-ref"], id_key, records)
        if mapped:
            args[param] = mapped
            continue
        # Nếu app của bảng x-ref chưa từng được gọi read tool -> chặn lại nhắc gọi read tool trước
        ref_app = spec["x-ref"].split(".", 1)[0]
        if ref_app not in called_apps:
            return f"Tham số '{param}={val}' chưa được tra cứu ({desc}). Hãy gọi tool đọc của '{ref_app}' để lấy ID thực tế trước."
    return None

