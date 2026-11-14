"""Vòng lặp ReAct:
  LLM sinh 1 lệnh -> ép kiểu & sửa x-ref (Tên -> ID, nguồn tiền) -> kiểm tra x-ref & env.validate (không ghi log)
  -> env.call -> đưa kết quả thật lại cho LLM.

Quy tắc an toàn (khớp zalo-ý-tưởng-final.txt, mục VI.1):
- Mọi lệnh phải qua verify_and_repair_xrefs + env.validate trước; lệnh sai schema hoặc đoán mò ID
  khi chưa gọi read tool KHÔNG được gọi vào env (không tính U).
- Không retry bằng env.call. Nếu validate lỗi, báo lỗi cho LLM sửa tối đa `max_repairs` lần.
- Không gọi lại đúng lệnh read đã gọi (tránh lặp vô tận, tránh tăng U).
- Dừng khi LLM trả <end>, sau sys.ask_missing_info, hết số bước hoặc hết thời gian.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

from .parser import clean_arguments, parse_call
from .prompt import Catalog, build_messages
from .xref import verify_and_repair_xrefs


_READ_CAP_STRICT_TOOLS = ("flights.search", "train_bus_tickets.search", "public_transit.find_routes")

_EMPTY_SEARCH_KEYS = {
    "contacts.search": "contacts",
    "event_tickets.search": "events",
    "grocery.search": "products",
    "recipes.search": "recipes",
    "ecommerce.search": "products",
    "maps.search_place": "places",
    "library.search": "books",
    "food_delivery.search_restaurants": "restaurants",
    "medicine.search": "medicines",
    "music.search": "tracks",
    "notes.search": "notes",
    "vod.search": "items",
    "workouts.search": "workouts",
}


class ScriptedLLM:
    """LLM giả cho test: phát lại một danh sách lệnh cố định rồi báo <end>. Chỉ dùng để kiểm thử harness."""

    def __init__(self, calls: List[dict]):
        self.calls = calls

    def generate(self, messages: List[dict]) -> str:
        k = sum(1 for m in messages if m["role"] == "assistant")
        if k < len(self.calls):
            c = self.calls[k]
            return json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, ensure_ascii=False)
        return "<end>"


def run_agent(llm, env, catalog: Catalog, ctx: dict, user_text: str, history: Optional[List[dict]],
              max_steps: int = 8, max_repairs: int = 2, deadline: Optional[float] = None,
              few_shot_block: str = "", enable_xref_guard: bool = True) -> dict:
    steps: List[dict] = []        # các bước đã thực thi (đưa lại cho LLM)
    executed_counts: dict = {}
    repairs = 0
    stop = "end"
    is_scripted = isinstance(llm, ScriptedLLM)
    for _ in range(max_steps + max_repairs):
        if deadline is not None and time.time() > deadline:
            stop = "timeout"
            break
        if sum(1 for s in steps if "error" not in s) >= max_steps:
            stop = "max_steps"
            break
        out = llm.generate(build_messages(catalog, ctx, user_text, history, steps,
                                          few_shot_block=few_shot_block))
        call = parse_call(out)
        if call is None:
            stop = "end"
            break
        spec = catalog.tools.get(call["name"])
        call["arguments"] = clean_arguments(call["arguments"], spec, user_text=user_text)
        if enable_xref_guard:
            xref_err = verify_and_repair_xrefs(
                call, spec, history, steps,
                user_text=user_text, ctx=ctx, strict_seen_ids=not is_scripted
            )
            if xref_err:
                repairs += 1
                if repairs > max_repairs:
                    stop = "xref_error"
                    break
                steps.append({"call": call, "error": xref_err})
                continue
        key = json.dumps(call, ensure_ascii=False, sort_keys=True)
        kind = (spec or {}).get("kind", "write")
        cname = call["name"]
        # Read tool gọi trùng tham số là lặp vô ích; write tool cho phép gọi lặp khi thực sự đổi điểm/voucher (như loyalty.redeem)
        max_dup = 1 if kind == "read" else 3
        if executed_counts.get(key, 0) >= max_dup:
            stop = "duplicate"
            break
        # Giới hạn số lần lặp cùng một hàm đọc: chặt (2) cho tìm chuyến đi, nới (7) cho truy vấn theo chuỗi ngày/nguyên liệu
        max_same_read = 2 if cname in _READ_CAP_STRICT_TOOLS else 7
        same_read_count = sum(1 for s in steps if s.get("call", {}).get("name") == cname and "error" not in s)
        if kind == "read" and same_read_count >= max_same_read:
            steps.append({"call": call, "error": f"Tool '{cname}' đã được gọi {same_read_count} lần. Hãy dùng kết quả đã có để gọi tool tiếp theo hoặc trả <end>."})
            repairs += 1
            if repairs > max_repairs:
                stop = "duplicate"
                break
            continue
        v = env.validate(call["name"], call["arguments"])
        if not v.get("ok"):
            repairs += 1
            if repairs > max_repairs:
                stop = "schema_error"
                break
            steps.append({"call": call, "error": str(v.get("error"))})
            continue
        res = env.call(call["name"], call["arguments"])
        if isinstance(res, dict) and res.get("done") is False:
            executed_counts[key] = max_dup
        else:
            executed_counts[key] = executed_counts.get(key, 0) + 1
        res_for_llm = res
        empty_key = _EMPTY_SEARCH_KEYS.get(cname)
        if (
            not is_scripted
            and empty_key is not None
            and isinstance(res, dict)
            and res.get(empty_key) == []
            and str((call.get("arguments") or {}).get("query") or "").strip() != ""
        ):
            res_for_llm = {
                **res,
                "_hint": "Tìm với từ khóa vừa rồi trả về danh sách rỗng (có thể do lỗi nghe ASR hoặc từ khóa quá chi tiết). Hãy gọi lại hàm tìm kiếm với query=\"\" (chuỗi rỗng) để xem toàn bộ danh sách rồi chọn mục phù hợp nhất!",
            }
        steps.append({"call": call, "result": res_for_llm})
        if call["name"] == "sys.ask_missing_info":
            stop = "ask_missing_info"
            break
    return {"stop": stop, "steps": steps}
