"""Dựng prompt cho LLM agent: schema rút gọn, ngữ cảnh thời gian, từ điển phương ngữ, lịch sử.

Dùng CHUNG cho huấn luyện (training/build_sft_data.py) và suy luận (agent.py), nên
định dạng ở đây là hợp đồng giữa hai phía: đổi ở đây thì phải huấn luyện lại LLM.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

END = "<end>"
RESULT_PREFIX = "KẾT QUẢ: "
ERROR_PREFIX = "LỖI SCHEMA (chưa thực thi, không tính vào log): "

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

# Từ điển phương ngữ / từ lóng / cách nói số-giờ: tri thức ngôn ngữ chung, KHÔNG gán đáp án theo mẫu nào.
GLOSSARY = (
    "Phương ngữ & từ lóng thường gặp (quy về tiếng Việt chuẩn trước khi hiểu): "
    "Trung: rứa/ri=vậy/thế này, ni=này, nớ=đó/ấy, mô=đâu, tê=kia, chi=gì, răng=sao/thế nào, "
    "bữa ni=hôm nay, hắn=anh/chị ấy, o=cô, mự=mợ. "
    "Nam: dùm/giùm=giúp, nghen/nghe/hen/nè=nhé, tui=tôi, hông=không, mần=làm, kêu=gọi, vô=vào, "
    "bữa nay=hôm nay, xíu=một chút, hổng=không. "
    "Bắc: ơi/nhé/đi/với=đề nghị lịch sự, ngoài ni=ngoài này. "
    "Tiền: ngàn/nghìn/k=1.000, chục=10, trăm=100, triệu/tr/củ=1.000.000, 'hai triệu tám'=2.800.000, "
    "'một triệu hai'=1.200.000. "
    "Giờ: rưỡi=+30 phút, 'tám giờ kém 15'=7:45, chiều/tối=PM, sáng=AM, trưa=~12h, "
    "mai=ngày mai, mốt/ngày kia=hôm nay+2, hôm kia=hôm nay-2, tuần sau=tuần kế tiếp. "
    "Viết tắt: ib=nhắn tin, ck=chuyển khoản, ĐH=điều hòa, AC=máy lạnh/điều hòa."
)

SYSTEM_RULES = (
    "Bạn là LifeMate, trợ lý giọng nói tiếng Việt chạy trên thiết bị. Bạn hoàn thành yêu cầu của người dùng "
    "bằng cách gọi tool, MỖI LƯỢT CHỈ MỘT LỆNH, ở dạng JSON một dòng: "
    '{"name":"app.tool","arguments":{...}}. Khi đã xong, trả đúng chuỗi ' + END + ".\n"
    "Quy tắc:\n"
    "1. Chỉ dùng tool trong danh sách bên dưới. Tham số có dấu * là bắt buộc; id<...> là ID phải lấy từ kết quả "
    "tool đọc (read) trước khi gọi tool ghi (write) — không bao giờ đoán ID.\n"
    "2. Chỉ gọi tool đọc khi cần ID hoặc dữ kiện để ra quyết định; các hàm liệt kê (.list, list_events, order_status) "
    "ưu tiên gọi với {} để xem đầy đủ danh sách trừ khi cần lọc rõ ràng.\n"
    "3. Thiếu tham số bắt buộc mà ngữ cảnh/profile/kết quả tool/lịch sử không suy ra được: mới gọi "
    'sys.ask_missing_info với {"tool_name":"app.tool","params":["tên_tham_số",...]} rồi dừng. '
    "NẾU thông tin (số điện thoại, ngày giờ, số lượng, đối tượng) ĐÃ CÓ trong lịch sử hội thoại trước đó thì PHẢI DÙNG, "
    "tuyệt đối không hỏi lại. Nếu câu có nhiều yêu cầu, hãy thực hiện xong các yêu cầu đã đủ thông tin TRƯỚC, rồi mới gọi sys.ask_missing_info ở bước cuối cùng.\n"
    "4. Phản hồi câu hỏi trước đó: Nếu lượt cuối của Trợ lý trong lịch sử là câu hỏi (ví dụ hỏi xác nhận đặt vé, "
    "chọn kênh, số lượng, phương thức...), người dùng trả lời ngắn ('ừ', 'đặt đi', '2 hộp', 'Zalo') thì đó là "
    "đồng ý/chọn lựa — PHẢI gọi tiếp tool hành động tương ứng, TUYỆT ĐỐI KHÔNG trả " + END + " rỗng!\n"
    "5. Yêu cầu không thuộc app nào đã cài (hoặc ngoài khả năng của các tool): trả " + END + " ngay, không gọi tool nào.\n"
    "6. Thời gian dùng ISO-8601 đầy đủ giây và offset múi giờ (ví dụ 2026-11-11T05:30:00+07:00), tính từ 'Bây giờ'. "
    "Số tiền là số nguyên không dấu. Văn bản tự do (~) giữ nguyên lời người dùng, ngắn gọn.\n"
    "7. Người dùng có thể nói ngập ngừng/tự sửa ('à không, ...'): lấy ý cuối cùng. Câu điều kiện ('nếu mưa thì...', "
    "'nếu không đủ số dư thì...'): đọc dữ kiện bằng tool đọc rồi kiểm tra kỹ số dư/điều kiện mới quyết định.\n"
    "8. Sau mỗi tool bạn nhận 'KẾT QUẢ: ...'. Dựa vào kết quả thật để gọi lệnh tiếp theo."
)


# ----------------------------------------------------------------------------- schema rút gọn
_TYPE = {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "array": "list", "object": "obj"}


_BOILERPLATE_DESC = {
    "Số nguyên, không dấu chấm hay phẩy, ví dụ 2": "",
    "Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000": "",
    "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00": "ISO-datetime",
    "YYYY-MM-DD, ví dụ 2026-11-29": "YYYY-MM-DD",
    "Giữ nguyên chuỗi, ví dụ 09123456785": "",
    "Số điện thoại, giữ nguyên chuỗi, ví dụ 09123456785": "số ĐT",
}


def _param_str(name: str, spec: dict, required: bool) -> str:
    t = spec.get("type", "string")
    desc = (spec.get("description") or "").strip()
    short_desc = _BOILERPLATE_DESC.get(desc, desc)
    if "enum" in spec:
        body = "|".join(str(x) for x in spec["enum"])
    elif "x-ref" in spec:
        body = f"id<{spec['x-ref']}>"
    elif short_desc in ("ISO-datetime", "YYYY-MM-DD"):
        body = short_desc
        short_desc = ""
    else:
        body = _TYPE.get(t, str(t))
        if t == "array":
            item_spec = spec.get("items", {})
            if item_spec.get("type") == "object" and "properties" in item_spec:
                ireq = set(item_spec.get("required", []))
                iparts = [_param_str(inm, isp, inm in ireq) for inm, isp in item_spec["properties"].items()]
                body = "list[{" + ", ".join(iparts) + "}]"
            else:
                body = f"list[{_TYPE.get(item_spec.get('type', 'string'), 'str')}]"
        if spec.get("format"):
            body += f"({spec['format']})"
    if spec.get("x-match") == "fuzzy":
        body += "~"
    if "exclusiveMinimum" in spec:
        body += ">0"
    out = f"{name}{'*' if required else ''}:{body}"
    if short_desc:
        # Bỏ phần boilerplate lặp lại cuối câu nếu có
        for tail in (". Số nguyên, không dấu chấm hay phẩy, ví dụ 2",
                     ", số nguyên, không dấu chấm hay phẩy, ví dụ 2",
                     "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"):
            short_desc = short_desc.replace(tail, "").strip(" ,;")
        if short_desc:
            out += f" [{short_desc}]"
    return out


def compact_tool(tool: dict) -> str:
    p = tool.get("parameters", {})
    req = set(p.get("required", []))
    parts = [_param_str(n, s, n in req) for n, s in p.get("properties", {}).items()]
    one_of = p.get("x-one-of")
    line = f"{tool['name']}({', '.join(parts)})"
    if one_of:
        groups = []
        for g in one_of:
            groups.append("+".join(g) if isinstance(g, list) else str(g))
        line += " [một trong: " + " | ".join(groups) + "]"
    desc = (tool.get("description") or "").strip()
    return f"{line} — {desc}" if desc else line


class Catalog:
    """Catalog của một split: tra schema theo tên tool, dựng khối tool cho từng sample."""

    def __init__(self, raw: dict):
        self.tools: Dict[str, dict] = {t["name"]: t for a in raw["apps"] for t in a["tools"]}
        self.apps: Dict[str, dict] = {a["id"]: a for a in raw["apps"]}
        self._cache: Dict[tuple, str] = {}

    def tools_block(self, installed_apps: dict) -> str:
        key = tuple(sorted((a, tuple(v.get("tools", []))) for a, v in installed_apps.items()))
        if key in self._cache:
            return self._cache[key]
        lines = []
        for app in sorted(installed_apps):
            names = installed_apps[app].get("tools", [])
            meta = self.apps.get(app, {})
            title = f"## {app}" + (f" ({meta['name_vi']})" if meta.get("name_vi") else "")
            lines.append(title)
            for n in names:
                if n in self.tools:
                    lines.append(compact_tool(self.tools[n]))
        if "sys.ask_missing_info" in self.tools:
            lines.append("## sys")
            lines.append(compact_tool(self.tools["sys.ask_missing_info"]))
        out = "\n".join(lines)
        self._cache[key] = out
        return out

    def fuzzy_params(self) -> Dict[str, set]:
        return {n: {p for p, s in t.get("parameters", {}).get("properties", {}).items()
                    if s.get("x-match") == "fuzzy"} for n, t in self.tools.items()}


# ----------------------------------------------------------------------------- thời gian & ngữ cảnh
def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def time_block(current_time: str, timezone: str = "") -> str:
    now = parse_time(current_time)
    off = current_time[-6:] if len(current_time) >= 6 and current_time[-6] in "+-" else ""
    head = f"Bây giờ: {current_time} ({WEEKDAYS[now.weekday()]}, múi giờ {timezone or off}, offset {off})."
    names = {0: "hôm nay", 1: "mai", 2: "ngày kia"}
    days = []
    for i in range(15):
        d = now + timedelta(days=i)
        days.append(f"{names.get(i, '')+' ' if i in names else ''}{WEEKDAYS[d.weekday()]} {d.date().isoformat()}")
    mon = now + timedelta(days=7 - now.weekday())
    nxt = f"Tuần sau bắt đầu {WEEKDAYS[0]} {mon.date().isoformat()}."
    return head + "\nLịch 15 ngày tới: " + " | ".join(days) + "\n" + nxt


def context_block(ctx: dict) -> str:
    prof = ctx.get("user_profile") or {}
    bits = [f"{k}={v}" for k, v in prof.items() if v]
    loc = ctx.get("location")
    if loc:
        bits.append(f"vị trí(lat,lng)={loc.get('lat')},{loc.get('lng')}")
    return "Hồ sơ người dùng: " + "; ".join(bits) if bits else "Hồ sơ người dùng: (trống)"


# ----------------------------------------------------------------------------- kết quả & lịch sử
def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def compact_result(res, limit: int = 2600) -> str:
    s = compact_json(res)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def call_json(name: str, arguments: dict) -> str:
    return compact_json({"name": name, "arguments": arguments})


def history_block(history: List[dict], result_limit: int = 2500) -> str:
    """history: [{"role":"user","text"}, {"role":"assistant","text","tool_calls","tool_results"}]"""
    if not history:
        return ""
    lines = ["Lịch sử hội thoại trước đó (các thao tác này ĐÃ được thực hiện):"]
    for t in history:
        if t.get("role") == "user":
            txt = (t.get("text") or "").strip()
            if txt:
                lines.append(f"Người dùng: {txt}")
        else:
            seg = f"Trợ lý: {t.get('text', '')}".rstrip()
            for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or []):
                seg += f"\n  gọi {call_json(c['name'], c.get('arguments', {}))} → {compact_result(r, result_limit)}"
            lines.append(seg)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- ghép message
def system_message(catalog: Catalog, installed_apps: dict) -> str:
    return SYSTEM_RULES + "\n\n" + GLOSSARY + "\n\nDanh sách tool:\n" + catalog.tools_block(installed_apps)


def user_message(ctx: dict, user_text: str, history: Optional[List[dict]] = None,
                 few_shot_block: str = "") -> str:
    parts = [time_block(ctx.get("current_time", ""), ctx.get("timezone", "")), context_block(ctx)]
    if few_shot_block:
        parts.append(few_shot_block)
    h = history_block(history or [])
    if h:
        parts.append(h)
    parts.append(f"Yêu cầu hiện tại của người dùng: {user_text}")
    return "\n".join(parts)


def build_messages(catalog: Catalog, ctx: dict, user_text: str, history: Optional[List[dict]],
                   steps: List[dict], few_shot_block: str = "") -> List[dict]:
    """steps: [{"call": {name, arguments}, "result": any}  hoặc  {"call":..., "error": str}]"""
    msgs = [{"role": "system", "content": system_message(catalog, ctx.get("installed_apps", {}))},
            {"role": "user", "content": user_message(ctx, user_text, history, few_shot_block=few_shot_block)}]
    for s in steps:
        msgs.append({"role": "assistant", "content": call_json(s["call"]["name"], s["call"].get("arguments", {}))})
        if "error" in s:
            msgs.append({"role": "user", "content": ERROR_PREFIX + s["error"]})
        else:
            msgs.append({"role": "user", "content": RESULT_PREFIX + compact_result(s["result"])})
    return msgs
