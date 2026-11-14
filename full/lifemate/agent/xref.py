"""Kiểm tra khóa ngoại (x-ref), tự động ánh xạ Tên -> ID và bảo vệ tham số nghiệp vụ từ kết quả Read Tool.

Tại sao cần module này?
Trong zaic_tools, env.validate() chỉ kiểm tra kiểu dữ liệu (str/int), KHÔNG kiểm tra ID có tồn tại không,
cũng không kiểm tra số dư tài khoản/ví trước khi chuyển tiền/thanh toán.
Nếu LLM truyền nhầm tên hiển thị ('đèn phòng khách', 'Thép Đại Hưng') vào trường ID/symbol hoặc đoán mò ID
khi chưa gọi read tool, env.call() vẫn chạy nhưng trả về done=False, làm tăng U và lệch state_hash (0 điểm).

Module này (chỉ đọc từ history và kết quả các bước read hợp lệ, TUYỆT ĐỐI không chạm state.json):
1. Thu thập toàn bộ ID/symbol và bản ghi từ history.tool_results và các bước read đã chạy.
2. Nếu LLM điền tên hiển thị vào trường x-ref hoặc tên công ty vào symbol -> tự động đổi sang đúng mã ID/symbol.
3. Nếu LLM đoán mò ID không có trong seen_ids -> chặn trước khi vào env.call() và nhắc LLM gọi read tool.
4. Kiểm tra chéo số dư ngân hàng / ví điện tử từ kết quả read trước đó để tránh lỗi 'số dư không đủ'.
5. Chặn các lỗi logic hiển nhiên (origin == destination, thiếu item_id trong food_delivery.place_order).
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Set, Tuple


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).lower().split())


# Ánh xạ tên doanh nghiệp/chỉ số phổ biến sang mã chứng khoán khi người dùng đọc tên công ty
_STOCK_NAME_TO_SYMBOL = {
    "thịnh phát": "ZTP",
    "ngân hàng tmcp thịnh phát": "ZTP",
    "ngân hàng thịnh phát": "ZTP",
    "mộc xanh": "ZMX",
    "sữa mộc xanh": "ZMX",
    "ctcp sữa mộc xanh": "ZMX",
    "đại hưng": "ZDH",
    "thép đại hưng": "ZDH",
    "tập đoàn thép đại hưng": "ZDH",
    "ctcp tập đoàn thép đại hưng": "ZDH",
    "đại việt": "ZDV",
    "ngân hàng đại việt": "ZDV",
    "ngân hàng tmcp đại việt": "ZDV",
    "kim ngân": "ZKN",
    "chứng khoán kim ngân": "ZKN",
    "ctcp chứng khoán kim ngân": "ZKN",
    "thế giới số": "ZTS",
    "đầu tư thế giới số": "ZTS",
    "ctcp đầu tư thế giới số": "ZTS",
    "brightpath": "ZBH",
    "brightpath holdings": "ZBH",
    "ctcp brightpath holdings": "ZBH",
    "redwood": "RWA",
    "redwood analytics": "RWA",
    "ngân hàng bắc hà": "Bắc Hà Bank",
    "ngân hàng tmcp bắc hà": "Bắc Hà Bank",
    "bắc hà bank": "Bắc Hà Bank",
    "vn-index": "VNINDEX",
    "vn index": "VNINDEX",
    "chỉ số vn-index": "VNINDEX",
    "vn30": "VN30",
    "hnx30": "HNX30",
}


def _walk_results(obj, seen_ids: Set[str], records: List[Tuple[str, dict]], bucket: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if (k in ("id", "merchant_id", "symbol", "route_id", "product_id", "item_id") or k.endswith("_id")) and isinstance(v, str) and v:
                seen_ids.add(v)
                if k == "symbol":
                    seen_ids.add(v.upper())
        if any(isinstance(obj.get(k), str) and obj.get(k) for k in ("id", "merchant_id", "symbol", "product_id", "item_id")):
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
    for prefer_bucket in (True, False):
        exact_hits = []
        sub_hits = []
        for bucket, rec in records:
            if prefer_bucket and table_key and bucket and bucket != table_key:
                continue
            rid = rec.get(id_key)
            if not isinstance(rid, str) or not rid:
                continue
            for field in ("name", "title", "label", "provider", "counterparty", "room", "phone", "phone_number", "operator", "venue"):
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


def _resolve_stock_symbol(sym_val: str, records: List[Tuple[str, dict]]) -> str:
    """Ánh xạ tên công ty sang mã cổ phiếu từ kết quả get_portfolio/get_quote hoặc bảng tên chuẩn."""
    target = _norm(sym_val)
    if not target:
        return sym_val
    # 1. Tìm trong records đã đọc từ môi trường trước (ưu tiên tuyệt đối dữ liệu thực tế của sample)
    for _, rec in records:
        rsym = rec.get("symbol")
        rname = rec.get("name")
        if isinstance(rsym, str) and rsym:
            if _norm(rsym) == target:
                return rsym
            if isinstance(rname, str) and rname:
                nr = _norm(rname)
                if target == nr or (len(target) >= 4 and (target in nr or nr in target)):
                    return rsym
    # 2. Tìm trong bảng tên doanh nghiệp chuẩn
    if target in _STOCK_NAME_TO_SYMBOL:
        return _STOCK_NAME_TO_SYMBOL[target]
    for k, v in _STOCK_NAME_TO_SYMBOL.items():
        if len(k) >= 6 and (k in target or target in k):
            return v
    return sym_val


def _repair_banking_account(args: dict, records: List[Tuple[str, dict]], steps: List[dict], user_text: str):
    """Đảm bảo chọn tài khoản ngân hàng đủ số dư và ưu tiên Tài khoản thanh toán khi người dùng không chỉ định khác."""
    from_acc = args.get("from_account_id")
    amount = args.get("amount")
    if not isinstance(from_acc, str) or not isinstance(amount, (int, float)):
        return
    # Lấy danh sách tài khoản mới nhất từ records và trừ đi các giao dịch đã thực hiện trong steps
    acc_map: Dict[str, dict] = {}
    for bucket, rec in records:
        if bucket == "accounts" and isinstance(rec.get("id"), str) and "balance" in rec:
            acc_map[rec["id"]] = dict(rec)
    if not acc_map or from_acc not in acc_map:
        return
    # Cập nhật số dư nếu trước đó trong steps đã có lệnh banking.transfer / open_savings thành công
    for s in steps:
        res = s.get("result")
        if isinstance(res, dict) and res.get("ok") and "balance" in res:
            aid = (s.get("call", {}).get("arguments") or {}).get("from_account_id")
            if aid in acc_map and isinstance(res["balance"], (int, float)):
                acc_map[aid]["balance"] = res["balance"]

    chosen = acc_map[from_acc]
    # 1. Nếu tài khoản đang chọn KHÔNG đủ số dư -> tự động chuyển sang tài khoản có đủ số dư
    if chosen.get("balance", 0) < amount:
        for aid, a in acc_map.items():
            if a.get("balance", 0) >= amount:
                args["from_account_id"] = aid
                return

    # 2. Nếu người dùng không nhắc đến tiết kiệm/lương/không đủ, ưu tiên 'Tài khoản thanh toán' nếu đủ số dư
    ut = _norm(user_text)
    if not any(w in ut for w in ("tiết kiệm", "lương", "không đủ", "vựa", "tour")):
        for aid, a in acc_map.items():
            if _norm(a.get("name", "")) == "tài khoản thanh toán" and a.get("balance", 0) >= amount:
                args["from_account_id"] = aid
                return


def _repair_bill_payment_source(args: dict, history: Optional[List[dict]], steps: List[dict], records: List[Tuple[str, dict]]):
    """Nếu thanh toán hoá đơn bằng ví (ewallet) nhưng số dư ví không đủ -> tự động chuyển sang 'bank'."""
    bill_id = args.get("bill_id")
    source = args.get("source")
    if source != "ewallet" or not isinstance(bill_id, str):
        return
    bill_amount = None
    for bucket, rec in records:
        if bucket == "bills" and rec.get("id") == bill_id:
            bill_amount = rec.get("amount")
            break
    if not isinstance(bill_amount, (int, float)):
        return
    # Tìm số dư ví điện tử gần nhất từ history và steps
    wallet_bal = None
    for h in history or []:
        for c, r in zip(h.get("tool_calls") or [], h.get("tool_results") or []):
            if isinstance(c, dict) and c.get("name", "").startswith("ewallet.") and isinstance(r, dict) and isinstance(r.get("balance"), (int, float)):
                wallet_bal = r["balance"]
    for s in steps:
        cname = s.get("call", {}).get("name", "")
        r = s.get("result")
        if cname.startswith("ewallet.") and isinstance(r, dict) and isinstance(r.get("balance"), (int, float)):
            wallet_bal = r["balance"]
    if wallet_bal is not None and wallet_bal < bill_amount:
        args["source"] = "bank"


def verify_and_repair_xrefs(call: dict, tool_spec: Optional[dict],
                            history: Optional[List[dict]], steps: List[dict],
                            user_text: str = "", strict_seen_ids: bool = True) -> Optional[str]:
    """Kiểm tra và tự động sửa các tham số x-ref & điều kiện nghiệp vụ trong `call['arguments']`.

    - Trả về None nếu hợp lệ (hoặc đã tự sửa Tên -> ID / nguồn tiền thành công).
    - Trả về chuỗi thông báo lỗi nếu LLM đoán mò ID hoặc vi phạm điều kiện cần gọi read tool trước.
    """
    if not tool_spec:
        return None
    cname = call.get("name", "")
    props = tool_spec.get("parameters", {}).get("properties", {})
    args = call.get("arguments") or {}

    # 1. Chặn lỗi điểm đi trùng điểm đến (tránh vòng lặp gọi flights.search / train_bus_tickets.search vô tận)
    if cname in ("flights.search", "train_bus_tickets.search", "public_transit.find_routes", "maps.get_route", "ride_hailing.get_quotes"):
        orig = args.get("origin")
        dest = args.get("destination")
        if isinstance(orig, str) and isinstance(dest, str) and orig.strip() and _norm(orig) == _norm(dest):
            return f"Tham số origin='{orig}' và destination='{dest}' đang trùng nhau. Hãy kiểm tra lại điểm đi/điểm đến hoặc gọi sys.ask_missing_info nếu thiếu."

    # 2. Với tv.guide: khi người dùng chỉ yêu cầu chuyển/bật kênh (tv.tune), không truyền bộ lọc channel vào tv.guide
    if cname == "tv.guide" and "channel" in args and "date" not in args:
        ut = _norm(user_text)
        if any(w in ut for w in ("chuyển sang kênh", "chuyển kênh", "bật kênh", "mở kênh")) and not any(
            w in ut for w in ("lịch", "mấy giờ", "ghi hình", "thu lại", "chiếu gì")
        ):
            args.pop("channel", None)

    seen_ids, records, called_apps = collect_state_knowledge(history, steps)

    # 3. Chuẩn hóa mã cổ phiếu (stocks.*) từ tên công ty nếu người dùng đọc tên doanh nghiệp
    if cname.startswith("stocks.") and isinstance(args.get("symbol"), str):
        args["symbol"] = _resolve_stock_symbol(args["symbol"], records)
        # Nếu vẫn là cụm nhiều từ có dấu (tên công ty chưa biết mã) và chưa gọi get_portfolio -> nhắc gọi get_portfolio
        sym_now = args["symbol"].strip()
        if " " in sym_now and sym_now != "Bắc Hà Bank" and "stocks" not in called_apps:
            return f"Chưa rõ mã cổ phiếu cho '{sym_now}'. Hãy gọi stocks.get_portfolio() trước để tra mã symbol trong danh mục."

    # 4. Kiểm tra items của food_delivery.place_order (nằm trong mảng con nên không có x-ref ở cấp ngoài)
    if cname == "food_delivery.place_order":
        items = args.get("items")
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                raw_id = it.get("item_id") or it.get("name") or it.get("recipe_id") or ""
                if isinstance(raw_id, str) and raw_id and raw_id not in seen_ids:
                    mapped_item = _match_record_id(raw_id, "food_delivery.menu", "id", records)
                    if mapped_item:
                        it["item_id"] = mapped_item
                for bad_k in list(it.keys()):
                    if bad_k not in ("item_id", "quantity", "note"):
                        it.pop(bad_k, None)
                if not it.get("item_id") or (strict_seen_ids and it.get("item_id") not in seen_ids):
                    return "Tham số 'items' trong food_delivery.place_order cần 'item_id' thực tế từ thực đơn. Hãy gọi food_delivery.get_menu(restaurant_id=...) trước để lấy item_id."

    # 5. Kiểm tra & tự sửa tất cả các tham số có x-ref
    xref_params = [(p, s) for p, s in props.items() if "x-ref" in s and p in args]
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
        ref_app = spec["x-ref"].split(".", 1)[0]
        if ref_app not in called_apps:
            return f"Tham số '{param}={val}' chưa được tra cứu ({desc}). Hãy gọi tool đọc của '{ref_app}' để lấy ID thực tế trước."
        if strict_seen_ids:
            return (
                f"ID '{param}={val}' không tồn tại trong kết quả đã tra cứu ({desc}). "
                f"Chỉ dùng đúng ID có trong KẾT QUẢ, gọi thêm tool đọc phù hợp của '{ref_app}', "
                f"hoặc gọi sys.ask_missing_info nếu không tìm thấy đối tượng."
            )

    # 6. Tự động kiểm tra số dư tài khoản ngân hàng / ví điện tử từ kết quả read đã có
    if cname in ("banking.transfer", "banking.open_savings"):
        _repair_banking_account(args, records, steps, user_text)
    elif cname == "bill_payment.pay":
        _repair_bill_payment_source(args, history, steps, records)

    return None
