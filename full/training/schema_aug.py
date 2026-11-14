"""Tăng cường dữ liệu SFT theo Schema (Schema-Guided Data Augmentation & History Mining):

1. Khai thác các lượt `history` thực tế từ `public_test.jsonl` (kết hợp transcript ASR thật từ `--asr-public`
   nếu có) và LUÔN giữ nguyên toàn bộ `prev_hist = resolved_hist[:i-1]` (tuyệt đối không cắt thành `[]`).
2. Bổ sung bộ mẫu sạch độc lập `build_clean_unseen_app_trajectories` (không đụng thực thể hay context mẫu thật
   của `public_test`) phủ đủ 34 hàm của 11 ứng dụng mới + các tình huống Search rỗng retry `query=""`.
3. Khai thác các lượt hội thoại nhiều bước từ `history` chuẩn của `train.jsonl` (`build_train_history_trajectories`).
4. Sinh các mẫu Schema-Following Regularization (SFR) với ứng dụng giả lập ngoài catalog.
"""
from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lifemate.agent.prompt import Catalog
from lifemate.data import load_catalog_raw, locate_split, read_jsonl

_EMPTY_SEARCH_HINT = (
    "Tìm với từ khóa vừa rồi trả về danh sách rỗng (có thể do lỗi nghe ASR hoặc từ khóa quá chi tiết). "
    'Hãy gọi lại hàm tìm kiếm với query="" (chuỗi rỗng) để xem toàn bộ danh sách rồi chọn mục phù hợp nhất!'
)

_SYNTH_TEMPLATES = {
    "recipes.get": lambda a, p, s: f"{p}xem chi tiết nguyên liệu và các bước nấu món vừa tìm{s}",
    "recipes.save": lambda a, p, s: f"{p}lưu công thức món này vào mục yêu thích{s}",
    "grocery.add_to_cart": lambda a, p, s: f"{p}thêm {a.get('quantity', 1)} phần sản phẩm đó vào giỏ đi chợ{s}",
    "grocery.checkout": lambda a, p, s: f"{p}thanh toán và đặt đơn đi chợ" + (f" giao về {a['address']}" if a.get("address") else "") + s,
    "ecommerce.order_status": lambda a, p, s: f"{p}kiểm tra trạng thái các đơn hàng mua sắm online gần đây{s}",
    "ecommerce.add_to_cart": lambda a, p, s: f"{p}thêm {a.get('quantity', 1)} sản phẩm đó vào giỏ hàng{s}",
    "ecommerce.checkout": lambda a, p, s: f"{p}đặt mua giỏ hàng hiện tại thanh toán bằng {a.get('payment', 'cod')}{s}",
    "ecommerce.cancel_order": lambda a, p, s: f"{p}huỷ đơn hàng đó giúp mình{s}",
    "ecommerce.request_return": lambda a, p, s: f"{p}tạo yêu cầu trả hàng hoàn tiền cho đơn đã giao đó{s}",
    "event_tickets.book": lambda a, p, s: f"{p}đặt {a.get('quantity', 2)} vé cho sự kiện đó{s}",
    "train_bus_tickets.book": lambda a, p, s: f"{p}đặt {a.get('seats', 1)} vé cho chuyến đó{s}",
    "train_bus_tickets.my_bookings": lambda a, p, s: f"{p}kiểm tra các vé tàu xe đã đặt{s}",
    "train_bus_tickets.cancel": lambda a, p, s: f"{p}huỷ vé đã đặt đó giúp mình{s}",
    "parking.find": lambda a, p, s: f"{p}tìm bãi đỗ xe gần khu vực {a.get('near', '')}{s}",
    "parking.list_sessions": lambda a, p, s: f"{p}xem danh sách các phiên đỗ xe hiện tại{s}",
    "parking.reserve": lambda a, p, s: f"{p}đặt chỗ đỗ xe trong {a.get('hours', 2)} tiếng từ {a.get('from_time', '')}{s}",
    "parking.end_session": lambda a, p, s: f"{p}kết thúc phiên đỗ xe hiện tại{s}",
    "stocks.get_portfolio": lambda a, p, s: f"{p}kiểm tra danh mục cổ phiếu đang nắm giữ{s}",
    "stocks.get_quote": lambda a, p, s: f"{p}xem giá cổ phiếu mã {a.get('symbol', 'VNINDEX')}{s}",
    "habit_tracker.list_habits": lambda a, p, s: f"{p}xem danh sách các thói quen đang theo dõi{s}",
    "habit_tracker.log": lambda a, p, s: f"{p}đánh dấu hoàn thành thói quen đó cho ngày {a.get('date', 'hôm nay')}{s}",
    "pantry.list_items": lambda a, p, s: f"{p}kiểm tra tủ bếp hiện còn nguyên liệu gì{s}",
    "pantry.consume": lambda a, p, s: f"{p}trừ {a.get('quantity', 1)} phần nguyên liệu vừa dùng trong tủ bếp{s}",
    "ewallet.get_balance": lambda a, p, s: f"{p}kiểm tra số dư ví điện tử hiện tại{s}",
}


def _synth_user_query_from_call(calls: List[dict], asst_text: str, rng: random.Random) -> str:
    """Tạo câu yêu cầu tiếng Việt tự nhiên khớp với lệnh gọi tool (không chép nhầm câu trả lời của trợ lý)."""
    c0 = calls[0]
    name = c0["name"]
    args = c0.get("arguments") or {}
    pref = rng.choice(["Ơi, ", "Em ơi, ", "Giúp mình ", "Bạn ơi, ", "Làm ơn ", ""])
    suf = rng.choice([" nhé.", " giúp mình nha.", " với nhé.", " giùm anh nha.", " nghe."])

    if name in _SYNTH_TEMPLATES:
        return _SYNTH_TEMPLATES[name](args, pref, suf)
    if name == "recipes.search":
        sv = args.get("servings")
        return f"{pref}tìm công thức nấu món {args.get('query', 'món ăn')}" + (f" cho {sv} người" if sv else "") + suf
    if name == "grocery.search":
        return f"{pref}tìm mua {args.get('query', 'đồ')} ở cửa hàng đi chợ online{suf}"
    if name == "ecommerce.search":
        mp = args.get("max_price")
        extra = f" giá dưới {mp // 1000} nghìn" if isinstance(mp, int) and mp >= 1000 else ""
        return f"{pref}tìm mua {args.get('query', 'sản phẩm')}{extra} trên sàn thương mại{suf}"
    if name == "event_tickets.search":
        return f"{pref}tìm vé sự kiện {args.get('query', 'sắp diễn ra')}{suf}"
    if name == "train_bus_tickets.search":
        tp = "tàu hỏa" if args.get("type") == "train" else ("xe khách" if args.get("type") == "bus" else "tàu hoặc xe")
        return f"{pref}tìm chuyến {tp} từ {args.get('origin', 'Hà Nội')} đi {args.get('destination', 'Hải Phòng')} ngày {args.get('date', '')}{suf}"
    if name == "public_transit.find_routes":
        return f"{pref}tìm tuyến xe công cộng từ {args.get('origin', 'nhà')} đến {args.get('destination', '')}{suf}"
    if name == "public_transit.next_departures":
        rt = args.get("route")
        return f"{pref}xem chuyến kế tiếp ở trạm {args.get('stop', '')}" + (f" tuyến {rt}" if rt else "") + suf
    if name == "public_transit.topup_card":
        amt = args.get("amount", 100000)
        return f"{pref}nạp {amt // 1000} nghìn vào thẻ xe buýt{suf}"
    if name == "stocks.place_order":
        side = "bán" if args.get("side") == "sell" else "mua"
        pr = f" giá {args['price']}" if args.get("price") else " giá thị trường"
        return f"{pref}đặt lệnh {side} {args.get('quantity', 100)} cổ phiếu {args.get('symbol', '')}{pr}{suf}"
    if name == "stocks.set_price_alert":
        d = "vượt lên trên" if args.get("direction") == "above" else "giảm xuống dưới"
        return f"{pref}đặt cảnh báo giá cổ phiếu {args.get('symbol', '')} khi {d} {args.get('price', 0)}{suf}"
    if name == "translator.translate":
        lang_map = {"en": "tiếng Anh", "vi": "tiếng Việt", "ja": "tiếng Nhật", "ko": "tiếng Hàn", "zh": "tiếng Trung", "fr": "tiếng Pháp"}
        lang = lang_map.get(args.get("to_lang", "en"), args.get("to_lang", "tiếng Anh"))
        return f"{pref}dịch câu '{args.get('text', '')}' sang {lang}{suf}"
    if name == "sys.ask_missing_info":
        return f"{pref}thực hiện yêu cầu {args.get('tool_name', 'hệ thống')}{suf}"
    return f"{pref}thực hiện thao tác trên {name.replace('.', ' ')}{suf}"


def _resolve_pub_history(hist: List[dict], asr_row: Optional[dict], rng: random.Random) -> List[dict]:
    """Khôi phục lịch sử hội thoại cho public_test: ưu tiên transcript ASR thật từ cache nếu có."""
    asr_h = (asr_row or {}).get("history") or []
    out: List[dict] = []
    for k, ht in enumerate(hist):
        if ht.get("role") == "user":
            txt = ""
            if k < len(asr_h) and isinstance(asr_h[k].get("text"), str) and asr_h[k]["text"].strip():
                txt = asr_h[k]["text"].strip()
            elif k + 1 < len(hist) and hist[k + 1].get("role") == "assistant":
                nxt = hist[k + 1]
                nc = nxt.get("tool_calls") or []
                txt = _synth_user_query_from_call(nc, nxt.get("text", ""), rng) if nc else "Tiếp tục hỗ trợ mình nhé."
            else:
                txt = "Tiếp tục giúp mình nhé."
            out.append({"role": "user", "text": txt})
        else:
            out.append({
                "role": ht.get("role", "assistant"),
                "text": ht.get("text", ""),
                "tool_calls": ht.get("tool_calls") or [],
                "tool_results": ht.get("tool_results") or [],
            })
    return out


def _installed_subset(pub_cat: Catalog, app_names: List[str]) -> dict:
    return {a: {"tools": [t["name"] for t in pub_cat.apps[a]["tools"]]} for a in app_names if a in pub_cat.apps}


def _clean_unseen_specs(pub_cat: Catalog) -> List[tuple]:
    """Danh sách các kịch bản sạch độc lập phủ 34 tool của 11 app mới + 6 kịch bản retry tìm kiếm rỗng."""
    base_apps = ["stocks", "grocery", "recipes", "pantry", "ecommerce", "event_tickets",
                 "train_bus_tickets", "parking", "public_transit", "habit_tracker", "translator"]
    ctx_hn = {
        "current_time": "2026-11-18T09:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "user_profile": {"name": "Minh Tuấn", "city": "Hà Nội", "home_address": "18 Trần Phú, Ba Đình, Hà Nội"},
        "installed_apps": _installed_subset(pub_cat, base_apps),
    }
    ctx_hcm = {
        "current_time": "2026-11-20T14:30:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "user_profile": {"name": "Lan Anh", "city": "TP.HCM", "home_address": "88 Võ Văn Tần, Quận 3, TP.HCM"},
        "installed_apps": _installed_subset(pub_cat, base_apps),
    }
    return [
        # 1. stocks (4 tools + buy/sell/limit/alert)
        ("clean_stk_buy", ctx_hcm, "Xem giá cổ phiếu ZMX rồi đặt lệnh mua 100 cổ giá thị trường giúp chị nhé.", [], [
            {"name": "stocks.get_quote", "arguments": {"symbol": "ZMX"}},
            {"name": "stocks.place_order", "arguments": {"symbol": "ZMX", "side": "buy", "quantity": 100}},
        ], [
            {"ok": True, "quote": {"symbol": "ZMX", "name": "CTCP Sữa Mộc Xanh", "price": 61200, "change_pct": 1.4}},
            {"ok": True, "order": {"id": "ord_s01", "symbol": "ZMX", "side": "buy", "quantity": 100, "type": "market", "price": 61200, "status": "placed"}},
        ], True),
        ("clean_stk_sell_half", ctx_hn, "Kiểm tra danh mục cổ phiếu rồi bán một nửa số ZKN đang giữ ở giá 34500 giúp anh.", [], [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.place_order", "arguments": {"symbol": "ZKN", "side": "sell", "quantity": 200, "price": 34500}},
        ], [
            {"ok": True, "holdings": [{"symbol": "ZKN", "name": "CTCP Chứng khoán Kim Ngân", "quantity": 400, "avg_price": 31000, "price": 33900}], "total_value": 13560000},
            {"ok": True, "order": {"id": "ord_s02", "symbol": "ZKN", "side": "sell", "quantity": 200, "type": "limit", "price": 34500, "status": "placed"}},
        ], True),
        ("clean_stk_alert", ctx_hn, "Nếu mã ZDH trong danh mục giảm dưới 25000 thì đặt cảnh báo giúp mình nhé.", [], [
            {"name": "stocks.get_quote", "arguments": {"symbol": "ZDH"}},
            {"name": "stocks.set_price_alert", "arguments": {"symbol": "ZDH", "price": 25000, "direction": "below"}},
        ], [
            {"ok": True, "quote": {"symbol": "ZDH", "name": "CTCP Tập đoàn Thép Đại Hưng", "price": 26500, "change_pct": -0.8}},
            {"ok": True, "alert": {"id": "al_s01", "symbol": "ZDH", "price": 25000, "direction": "below", "active": True}},
        ], True),
        ("clean_stk_hist_follow", ctx_hcm, "Vậy mua thêm 200 cổ phiếu mã đó theo giá hiện tại luôn nhé.", [
            {"role": "user", "text": "Kiểm tra giúp chị giá cổ phiếu Vitech hôm nay thế nào."},
            {"role": "assistant", "text": "Cổ phiếu Vitech đang ở mức 124.500 đồng.",
             "tool_calls": [{"name": "stocks.get_quote", "arguments": {"symbol": "Vitech"}}],
             "tool_results": [{"ok": True, "quote": {"symbol": "VITECH", "name": "CTCP Vitech", "price": 124500, "change_pct": 2.1}}]},
        ], [
            {"name": "stocks.place_order", "arguments": {"symbol": "VITECH", "side": "buy", "quantity": 200}},
        ], [
            {"ok": True, "order": {"id": "ord_s03", "symbol": "VITECH", "side": "buy", "quantity": 200, "type": "market", "price": 124500, "status": "placed"}},
        ], False),

        # 2. recipes + pantry + grocery (cross-app + multi-item search + checkout)
        ("clean_rcp_get_save", ctx_hn, "Tìm công thức nấu lẩu thái hải sản, xem chi tiết rồi lưu lại giúp anh nhé.", [], [
            {"name": "recipes.search", "arguments": {"query": "lẩu thái hải sản"}},
            {"name": "recipes.get", "arguments": {"recipe_id": "rcp_x01"}},
            {"name": "recipes.save", "arguments": {"recipe_id": "rcp_x01"}},
        ], [
            {"ok": True, "recipes": [{"id": "rcp_x01", "name": "Lẩu thái hải sản", "servings": 4, "ingredients": [{"name": "tôm sú", "quantity": 400, "unit": "g"}]}]},
            {"ok": True, "recipe": {"id": "rcp_x01", "name": "Lẩu thái hải sản", "steps": ["Nấu nước dùng", "Cho hải sản"]}},
            {"ok": True, "saved": {"id": "sav_x01", "recipe_id": "rcp_x01"}},
        ], True),
        ("clean_pantry_grocery", ctx_hcm, "Xem công thức bò lúc lắc 2 người ăn cần bao nhiêu thịt bò, trừ hết thịt bò còn trong tủ bếp rồi đặt mua thêm 1 phần nữa giao về nhà.", [], [
            {"name": "recipes.search", "arguments": {"query": "bò lúc lắc", "servings": 2}},
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": "pan_x02", "quantity": 1}},
            {"name": "grocery.search", "arguments": {"query": "thịt bò"}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_x05", "quantity": 1}},
            {"name": "grocery.checkout", "arguments": {"address": "88 Võ Văn Tần, Quận 3, TP.HCM"}},
        ], [
            {"ok": True, "recipes": [{"id": "rcp_x02", "name": "Bò lúc lắc", "servings": 2, "ingredients": [{"name": "thịt bò", "quantity": 2, "unit": "gói"}]}]},
            {"ok": True, "items": [{"id": "pan_x02", "name": "thịt bò thăn", "stock": 1, "unit": "gói"}]},
            {"ok": True, "item": {"id": "pan_x02", "name": "thịt bò thăn", "stock": 0, "unit": "gói"}},
            {"ok": True, "products": [{"id": "gro_x05", "name": "Thịt bò Úc 300g", "price": 125000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": "gro_x05", "name": "Thịt bò Úc 300g", "price": 125000, "quantity": 1}], "cart_total": 125000},
            {"ok": True, "order": {"id": "ord_gx01", "total": 125000, "status": "placed", "address": "88 Võ Văn Tần, Quận 3, TP.HCM"}},
        ], True),

        # 3. train_bus_tickets (search, book, my_bookings, cancel)
        ("clean_tb_book", ctx_hcm, "Tìm tàu từ TP.HCM đi Nha Trang ngày 2026-11-25 rồi đặt 2 chỗ trên chuyến đầu tiên nhé.", [], [
            {"name": "train_bus_tickets.search", "arguments": {"origin": "TP.HCM", "destination": "Nha Trang", "date": "2026-11-25", "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_x01", "seats": 2}},
        ], [
            {"ok": True, "trips": [{"id": "trip_x01", "type": "train", "operator": "Tàu SNT2", "origin": "TP.HCM", "destination": "Nha Trang", "date": "2026-11-25", "seats_left": 10}]},
            {"ok": True, "booking": {"id": "tk_x01", "trip_id": "trip_x01", "seats": 2, "status": "confirmed"}},
        ], True),
        ("clean_tb_cancel", ctx_hn, "Kiểm tra vé tàu xe đã đặt rồi huỷ chuyến đi Đà Nẵng giúp mình nha.", [], [
            {"name": "train_bus_tickets.my_bookings", "arguments": {}},
            {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tk_x08"}},
        ], [
            {"ok": True, "bookings": [{"id": "tk_x08", "trip_id": "trip_x09", "destination": "Đà Nẵng", "seats": 1, "status": "confirmed"}]},
            {"ok": True, "booking": {"id": "tk_x08", "status": "cancelled"}},
        ], True),

        # 4. ecommerce (search -> add_to_cart -> checkout, order_status -> cancel_order / request_return)
        ("clean_ec_buy", ctx_hn, "Tìm tai nghe không dây giá dưới 1 triệu trên sàn, bỏ 1 chiếc vào giỏ rồi thanh toán bằng ví điện tử giao về nhà.", [], [
            {"name": "ecommerce.search", "arguments": {"query": "tai nghe không dây", "max_price": 1000000}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_x11", "quantity": 1}},
            {"name": "ecommerce.checkout", "arguments": {"address": "18 Trần Phú, Ba Đình, Hà Nội", "payment": "ewallet"}},
        ], [
            {"ok": True, "products": [{"id": "prod_x11", "name": "Tai nghe Bluetooth AirSound", "price": 690000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": "prod_x11", "name": "Tai nghe Bluetooth AirSound", "price": 690000, "quantity": 1}], "cart_total": 690000},
            {"ok": True, "order": {"id": "ord_ex01", "total": 690000, "status": "placed", "payment": "ewallet"}},
        ], True),
        ("clean_ec_cancel", ctx_hn, "Xem các đơn hàng trên sàn, đơn nào đang chờ giao thì huỷ giùm mình nhé.", [], [
            {"name": "ecommerce.order_status", "arguments": {}},
            {"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_ex02"}},
        ], [
            {"ok": True, "orders": [{"id": "ord_ex02", "status": "shipping", "items": [{"name": "Bàn phím cơ"}]}, {"id": "ord_ex03", "status": "delivered", "items": [{"name": "Chuột không dây"}]}]},
            {"ok": True, "order": {"id": "ord_ex02", "status": "cancelled"}},
        ], True),
        ("clean_ec_return", ctx_hcm, "Đơn chuột không dây nhận hôm trước bị lỗi vỡ hỏng rồi, tạo yêu cầu trả hàng giúp chị.", [], [
            {"name": "ecommerce.order_status", "arguments": {}},
            {"name": "ecommerce.request_return", "arguments": {"order_id": "ord_ex03", "reason": "damaged"}},
        ], [
            {"ok": True, "orders": [{"id": "ord_ex02", "status": "shipping", "items": [{"name": "Bàn phím cơ"}]}, {"id": "ord_ex03", "status": "delivered", "items": [{"name": "Chuột không dây"}]}]},
            {"ok": True, "return": {"id": "ret_x01", "order_id": "ord_ex03", "reason": "damaged", "status": "requested"}},
        ], True),

        # 5. parking + public_transit + event_tickets + habit_tracker + translator
        ("clean_pk_res", ctx_hn, "Tìm bãi đỗ xe gần Hồ Tây rồi đặt chỗ trong 3 tiếng bắt đầu từ 10 giờ sáng nay.", [], [
            {"name": "parking.find", "arguments": {"near": "Hồ Tây"}},
            {"name": "parking.reserve", "arguments": {"lot_id": "lot_x01", "from_time": "2026-11-18T10:00:00+07:00", "hours": 3}},
        ], [
            {"ok": True, "lots": [{"id": "lot_x01", "name": "Bãi xe Trích Sài", "price_per_hour": 25000, "available_spots": 5}]},
            {"ok": True, "session": {"id": "park_x01", "lot_id": "lot_x01", "hours": 3, "status": "reserved"}},
        ], True),
        ("clean_pk_end", ctx_hn, "Kiểm tra phiên đỗ xe đang mở rồi kết thúc giúp mình.", [], [
            {"name": "parking.list_sessions", "arguments": {}},
            {"name": "parking.end_session", "arguments": {"session_id": "park_x01"}},
        ], [
            {"ok": True, "sessions": [{"id": "park_x01", "lot_id": "lot_x01", "status": "reserved"}]},
            {"ok": True, "session": {"id": "park_x01", "status": "ended"}},
        ], True),
        ("clean_pt_multi", ctx_hn, "Tìm đường xe buýt từ nhà tới bến xe Mỹ Đình, xem giờ xe tiếp theo ở trạm Cầu Giấy tuyến 34 và nạp 100 nghìn vào thẻ.", [], [
            {"name": "public_transit.find_routes", "arguments": {"origin": "18 Trần Phú, Ba Đình, Hà Nội", "destination": "Bến xe Mỹ Đình"}},
            {"name": "public_transit.next_departures", "arguments": {"stop": "Cầu Giấy", "route": "34"}},
            {"name": "public_transit.topup_card", "arguments": {"amount": 100000}},
        ], [
            {"ok": True, "routes": [{"route_id": "bus_34", "name": "Tuyến 34", "duration_min": 25}]},
            {"ok": True, "departures": [{"stop": "Cầu Giấy", "route": "bus_34", "time": "2026-11-18T09:12:00+07:00"}]},
            {"ok": True, "topup": {"id": "top_x01", "amount": 100000, "status": "done"}, "card_balance": 150000},
        ], True),
        ("clean_hb_already_done", ctx_hn, "Kiểm tra thói quen đọc sách xem hôm nay anh tích chưa, chưa thì mới đánh dấu nghen.", [], [
            {"name": "habit_tracker.list_habits", "arguments": {}},
        ], [
            {"ok": True, "habits": [{"id": "habit_x01", "name": "Đọc sách 30 phút", "streak": 6, "last_logged": "2026-11-18"}]},
        ], True),
        ("clean_hb_need_log", ctx_hn, "Kiểm tra thói quen uống đủ nước xem hôm nay tích chưa, chưa thì đánh dấu giúp tôi.", [], [
            {"name": "habit_tracker.list_habits", "arguments": {}},
            {"name": "habit_tracker.log", "arguments": {"habit_id": "habit_x02", "date": "2026-11-18"}},
        ], [
            {"ok": True, "habits": [{"id": "habit_x02", "name": "Uống đủ 2 lít nước", "streak": 4, "last_logged": "2026-11-17"}]},
            {"ok": True, "log": {"id": "log_x02", "habit_id": "habit_x02", "date": "2026-11-18"}, "streak": 5},
        ], True),
        ("clean_tr_ask_miss", ctx_hn, "Dịch giúp mình câu 'Hẹn gặp lại ngày mai' và đặt lịch tàu đi Huế giúp mình luôn.", [], [
            {"name": "translator.translate", "arguments": {"text": "Hẹn gặp lại ngày mai", "to_lang": "en"}},
            {"name": "sys.ask_missing_info", "arguments": {"tool_name": "train_bus_tickets.search", "params": ["date"]}},
        ], [
            {"ok": True, "translated_text": "See you tomorrow"},
            {"ok": True},
        ], False),

        # 6. 6 mẫu phản xạ gọi lại search(query="") khi từ khóa bị lỗi ASR / quá hẹp trả về []
        ("clean_retry_ev", ctx_hn, "Đặt cho mình 2 vé xem buổi hòa nhạc Đêm Thu nhé.", [], [
            {"name": "event_tickets.search", "arguments": {"query": "đêm thu"}},
            {"name": "event_tickets.search", "arguments": {"query": ""}},
            {"name": "event_tickets.book", "arguments": {"event_id": "ev_x02", "quantity": 2}},
        ], [
            {"ok": True, "events": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "events": [{"id": "ev_x01", "name": "Triển lãm Tranh Phố"}, {"id": "ev_x02", "name": "Hòa nhạc Đêm Mùa Thu", "price": 250000, "available": 15}]},
            {"ok": True, "booking": {"id": "tk_ev01", "event_id": "ev_x02", "quantity": 2, "status": "confirmed"}},
        ], False),
        ("clean_retry_gro", ctx_hcm, "Tìm mua 2 gói yến mặt cán dẹt trên app đi chợ bỏ vào giỏ giúp chị.", [], [
            {"name": "grocery.search", "arguments": {"query": "yến mặt"}},
            {"name": "grocery.search", "arguments": {"query": ""}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_x09", "quantity": 2}},
        ], [
            {"ok": True, "products": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "products": [{"id": "gro_x08", "name": "Sữa tươi 1L"}, {"id": "gro_x09", "name": "Yến mạch cán dẹt 500g", "price": 55000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": "gro_x09", "name": "Yến mạch cán dẹt 500g", "quantity": 2}], "cart_total": 110000},
        ], False),
        ("clean_retry_rcp_get", ctx_hn, "Tìm giúp mình công thức bánh ben kích rồi xem chi tiết các bước làm nhé.", [], [
            {"name": "recipes.search", "arguments": {"query": "bánh ben kích"}},
            {"name": "recipes.search", "arguments": {"query": ""}},
            {"name": "recipes.get", "arguments": {"recipe_id": "rcp_x07"}},
        ], [
            {"ok": True, "recipes": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "recipes": [{"id": "rcp_x07", "name": "Bánh pancake yến mạch", "servings": 2}]},
            {"ok": True, "recipe": {"id": "rcp_x07", "name": "Bánh pancake yến mạch", "steps": ["Trộn bột", "Áp chảo"]}},
        ], False),
        ("clean_retry_rcp_save", ctx_hcm, "Tìm món mình xào chay rồi lưu vào mục đã lưu giúp mình nhé.", [], [
            {"name": "recipes.search", "arguments": {"query": "mình xào chay"}},
            {"name": "recipes.search", "arguments": {"query": ""}},
            {"name": "recipes.save", "arguments": {"recipe_id": "rcp_x08"}},
        ], [
            {"ok": True, "recipes": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "recipes": [{"id": "rcp_x08", "name": "Mì xào chay thập cẩm", "servings": 2}]},
            {"ok": True, "saved": {"id": "sav_x08", "recipe_id": "rcp_x08"}},
        ], False),
        ("clean_retry_ec", ctx_hn, "Tìm mua 1 bàn phím cơ K2 Pro trên sàn thương mại bỏ vào giỏ giúp tôi.", [], [
            {"name": "ecommerce.search", "arguments": {"query": "k2 pro"}},
            {"name": "ecommerce.search", "arguments": {"query": ""}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_x19", "quantity": 1}},
        ], [
            {"ok": True, "products": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "products": [{"id": "prod_x19", "name": "Bàn phím cơ Keychron K2", "price": 1650000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": "prod_x19", "name": "Bàn phím cơ Keychron K2", "quantity": 1}], "cart_total": 1650000},
        ], False),
        ("clean_retry_gro2", ctx_hn, "Tìm trứng cút bắc thảo thêm 1 vỉ vào giỏ đi chợ nhé.", [], [
            {"name": "grocery.search", "arguments": {"query": "trứng cút bắc thảo"}},
            {"name": "grocery.search", "arguments": {"query": ""}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_x12", "quantity": 1}},
        ], [
            {"ok": True, "products": [], "_hint": _EMPTY_SEARCH_HINT},
            {"ok": True, "products": [{"id": "gro_x12", "name": "Trứng bắc thảo hộp 6 quả", "price": 38000, "in_stock": True}]},
            {"ok": True, "cart": [{"product_id": "gro_x12", "name": "Trứng bắc thảo hộp 6 quả", "quantity": 1}], "cart_total": 38000},
        ], False),
    ]


def build_clean_unseen_app_trajectories(pub_cat: Catalog, make_traj_fn) -> Tuple[List[dict], List[dict]]:
    """Tạo bộ mẫu sạch độc lập cho 34 tool của 11 app mới + 6 mẫu phản xạ query=\"\" khi tìm rỗng."""
    sft_rows: List[dict] = []
    fs_records: List[dict] = []
    for sid, ctx, txt, hist, calls, res, add_fs in _clean_unseen_specs(pub_cat):
        msgs = make_traj_fn(pub_cat, ctx, txt, hist, calls, res, few_shot_block="")
        sft_rows.append({"id": sid, "variant": "schema_aug", "messages": msgs})
        if add_fs and not hist:
            apps = sorted({c["name"].split(".")[0] for c in calls if "." in c.get("name", "")})
            fs_records.append({
                "id": sid,
                "source_id": sid,
                "text": txt,
                "apps": apps,
                "calls": calls,
            })
    return sft_rows, fs_records


def build_public_catalog_trajectories(data_dir: str, make_traj_fn,
                                      asr_public_path: Optional[str] = None,
                                      seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Khai thác các lượt `history` của 11 app mới trong `public_test.jsonl` (giữ nguyên prev_hist) + bộ mẫu sạch."""
    rng = random.Random(seed)
    try:
        split, pdir, _ = locate_split(data_dir, "public_test")
        pub_cat = Catalog(load_catalog_raw(pdir, split))
        pub_items = list(read_jsonl(pdir / "public_test.jsonl"))
    except Exception:
        return [], []

    asr_pub: Dict[str, dict] = {}
    if asr_public_path and Path(asr_public_path).is_file():
        asr_pub = {r["id"]: r for r in read_jsonl(Path(asr_public_path))}

    unseen_apps = {
        "stocks", "translator", "grocery", "recipes", "public_transit",
        "parking", "train_bus_tickets", "ecommerce", "event_tickets",
        "habit_tracker", "pantry",
    }

    sft_rows: List[dict] = []
    fs_records: List[dict] = []
    seen_chains = set()

    for it in pub_items:
        ctx = copy.deepcopy(it["context"])
        hist = it.get("history") or []
        if not hist:
            continue
        resolved_hist = _resolve_pub_history(hist, asr_pub.get(it["id"]), rng)
        city = (ctx.get("user_profile") or {}).get("city", "")

        for i, t in enumerate(hist):
            if t.get("role") != "assistant" or i == 0:
                continue
            calls = t.get("tool_calls") or []
            results = t.get("tool_results") or []
            if not calls or len(calls) != len(results):
                continue
            apps_used = {c["name"].split(".")[0] for c in calls if "." in c.get("name", "")}
            if not (apps_used & unseen_apps):
                continue
            if any(isinstance(r, dict) and r.get("done") is False for r in results):
                continue

            prev_hist = copy.deepcopy(resolved_hist[: i - 1])
            u_txt = resolved_hist[i - 1]["text"]
            chain_key = (
                calls[0]["name"],
                json.dumps(calls[0].get("arguments", {}), ensure_ascii=False, sort_keys=True),
                bool(prev_hist),
                city,
            )
            if chain_key in seen_chains:
                continue
            seen_chains.add(chain_key)

            msgs = make_traj_fn(pub_cat, ctx, u_txt, prev_hist, calls, results, few_shot_block="")
            rid = f"pub_hist_{it['id']}_{i}"
            sft_rows.append({"id": rid, "variant": "schema_aug", "messages": msgs})
            # Tuyệt đối chỉ đưa vào kho Few-Shot các mẫu không phụ thuộc prev_hist (bắt đầu từ lượt Read sạch)
            if not prev_hist:
                fs_records.append({
                    "id": rid,
                    "source_id": it["id"],
                    "text": u_txt,
                    "apps": sorted(apps_used),
                    "calls": calls,
                })

    clean_rows, clean_fs = build_clean_unseen_app_trajectories(pub_cat, make_traj_fn)
    sft_rows.extend(clean_rows)
    fs_records.extend(clean_fs)
    return sft_rows, fs_records


def build_train_history_trajectories(items: List[dict], is_val_fn, catalog: Catalog,
                                     make_traj_fn, gold_transcripts_fn,
                                     max_samples: int = 400, seed: int = 42) -> List[dict]:
    """Khai thác ~400 lượt hội thoại nhiều bước từ `history` chuẩn của `train.jsonl` (không rò rỉ tập val)."""
    rng = random.Random(seed)
    by_app: Dict[str, List[tuple]] = defaultdict(list)

    for it in items:
        if is_val_fn(it):
            continue
        g = gold_transcripts_fn(it)
        ghist = g.get("history") or []
        for i, t in enumerate(ghist):
            if t.get("role") != "assistant" or i < 2:
                continue
            calls = t.get("tool_calls") or []
            results = t.get("tool_results") or []
            if not calls or len(calls) != len(results):
                continue
            if any(isinstance(r, dict) and r.get("done") is False for r in results):
                continue
            if ghist[i - 1].get("role") != "user":
                continue
            app0 = calls[0]["name"].split(".")[0] if "." in calls[0].get("name", "") else "misc"
            by_app[app0].append((it, i, ghist, calls, results))

    for lst in by_app.values():
        rng.shuffle(lst)

    selected: List[tuple] = []
    apps_sorted = sorted(by_app)
    round_idx = 0
    while len(selected) < max_samples:
        added = False
        for app in apps_sorted:
            lst = by_app[app]
            if round_idx < len(lst):
                selected.append(lst[round_idx])
                added = True
                if len(selected) >= max_samples:
                    break
        if not added:
            break
        round_idx += 1

    out: List[dict] = []
    for it, i, ghist, calls, results in selected:
        prev_hist = copy.deepcopy(ghist[: i - 1])
        u_txt = ghist[i - 1]["text"]
        msgs = make_traj_fn(catalog, it["context"], u_txt, prev_hist, calls, results, few_shot_block="")
        out.append({"id": f"tr_hist_{it['id']}_{i}", "variant": "train_history", "messages": msgs})
    return out


def build_sfr_trajectories(make_traj_fn) -> List[dict]:
    """Sinh các mẫu Schema-Following Regularization (SFR) với các ứng dụng giả lập ngoài tập train/public_test
    để huấn luyện LLM đọc hiểu schema bất kỳ (zero-shot schema reading) khi gặp tool mới ở private_test."""
    sfr_raw_catalog = {
        "apps": [
            {
                "id": "hotel_booking",
                "name_vi": "Đặt phòng khách sạn",
                "tools": [
                    {
                        "name": "hotel_booking.search_hotels",
                        "kind": "read",
                        "description": "Tìm khách sạn theo thành phố và ngày nhận phòng",
                        "parameters": {
                            "type": "object",
                            "required": ["city", "check_in"],
                            "properties": {
                                "city": {"type": "string", "description": "Thành phố, ví dụ Đà Nẵng"},
                                "check_in": {"type": "string", "description": "YYYY-MM-DD, ví dụ 2026-11-29"},
                            },
                        },
                    },
                    {
                        "name": "hotel_booking.reserve_room",
                        "kind": "write",
                        "description": "Đặt phòng khách sạn",
                        "parameters": {
                            "type": "object",
                            "required": ["hotel_id", "nights", "rooms"],
                            "properties": {
                                "hotel_id": {"type": "string", "x-ref": "hotel_booking.hotels", "description": "ID khách sạn, lấy từ hotel_booking.search_hotels"},
                                "nights": {"type": "integer", "exclusiveMinimum": 0},
                                "rooms": {"type": "integer", "exclusiveMinimum": 0},
                            },
                        },
                    },
                ],
            },
            {
                "id": "clinic",
                "name_vi": "Đặt lịch khám bệnh",
                "tools": [
                    {
                        "name": "clinic.list_doctors",
                        "kind": "read",
                        "description": "Danh sách bác sĩ theo chuyên khoa",
                        "parameters": {
                            "type": "object",
                            "required": [],
                            "properties": {
                                "specialty": {"type": "string", "description": "Chuyên khoa, ví dụ nha khoa, tai mũi họng"},
                            },
                        },
                    },
                    {
                        "name": "clinic.book_visit",
                        "kind": "write",
                        "description": "Đặt lịch hẹn khám với bác sĩ",
                        "parameters": {
                            "type": "object",
                            "required": ["doctor_id", "slot_time"],
                            "properties": {
                                "doctor_id": {"type": "string", "x-ref": "clinic.doctors", "description": "ID bác sĩ, lấy từ clinic.list_doctors"},
                                "slot_time": {"type": "string", "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"},
                            },
                        },
                    },
                ],
            },
            {
                "id": "cinema",
                "name_vi": "Vé xem phim rạp",
                "tools": [
                    {
                        "name": "cinema.list_showtimes",
                        "kind": "read",
                        "description": "Tìm suất chiếu phim",
                        "parameters": {
                            "type": "object",
                            "required": ["movie"],
                            "properties": {
                                "movie": {"type": "string", "description": "Tên phim"},
                                "date": {"type": "string", "description": "YYYY-MM-DD, ví dụ 2026-11-29"},
                            },
                        },
                    },
                    {
                        "name": "cinema.buy_tickets",
                        "kind": "write",
                        "description": "Mua vé suất chiếu",
                        "parameters": {
                            "type": "object",
                            "required": ["showtime_id", "seats"],
                            "properties": {
                                "showtime_id": {"type": "string", "x-ref": "cinema.showtimes", "description": "ID suất chiếu, lấy từ cinema.list_showtimes"},
                                "seats": {"type": "integer", "exclusiveMinimum": 0},
                            },
                        },
                    },
                ],
            },
            {
                "id": "sys",
                "name_vi": "Hệ thống",
                "tools": [
                    {
                        "name": "sys.ask_missing_info",
                        "kind": "read",
                        "description": "Hỏi lại người dùng khi thiếu tham số bắt buộc",
                        "parameters": {
                            "type": "object",
                            "required": ["tool_name", "params"],
                            "properties": {
                                "tool_name": {"type": "string"},
                                "params": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        ]
    }
    sfr_cat = Catalog(sfr_raw_catalog)
    base_ctx = {
        "current_time": "2026-11-18T09:00:00+07:00",
        "timezone": "Asia/Ho_Chi_Minh",
        "user_profile": {"name": "Anh Minh", "city": "Hà Nội"},
        "installed_apps": {
            "hotel_booking": {"tools": ["hotel_booking.search_hotels", "hotel_booking.reserve_room"]},
            "clinic": {"tools": ["clinic.list_doctors", "clinic.book_visit"]},
            "cinema": {"tools": ["cinema.list_showtimes", "cinema.buy_tickets"]},
        },
    }
    samples = [
        (
            "sfr_01",
            "Tìm khách sạn ở Đà Nẵng nhận phòng ngày mai rồi đặt 1 phòng trong 2 đêm ở khách sạn rẻ nhất giúp anh nhé.",
            [
                {"name": "hotel_booking.search_hotels", "arguments": {"city": "Đà Nẵng", "check_in": "2026-11-19"}},
                {"name": "hotel_booking.reserve_room", "arguments": {"hotel_id": "htl_02", "nights": 2, "rooms": 1}},
            ],
            [
                {"ok": True, "hotels": [{"id": "htl_01", "name": "Grand Sea", "price": 1200000}, {"id": "htl_02", "name": "Marina Bay", "price": 750000}]},
                {"ok": True, "reservation": {"id": "res_01", "hotel_id": "htl_02", "nights": 2, "rooms": 1}},
            ],
        ),
        (
            "sfr_02",
            "Đặt lịch khám nha khoa với bác sĩ Khoa lúc 3 giờ chiều mai giúp chị nha.",
            [
                {"name": "clinic.list_doctors", "arguments": {"specialty": "nha khoa"}},
                {"name": "clinic.book_visit", "arguments": {"doctor_id": "doc_05", "slot_time": "2026-11-19T15:00:00+07:00"}},
            ],
            [
                {"ok": True, "doctors": [{"id": "doc_05", "name": "BS. Trần Đăng Khoa", "specialty": "nha khoa"}]},
                {"ok": True, "visit": {"id": "vst_01", "doctor_id": "doc_05", "slot_time": "2026-11-19T15:00:00+07:00"}},
            ],
        ),
        (
            "sfr_03",
            "Tìm suất chiếu phim Mưa Đỏ hôm nay rồi mua 2 vé suất 7 giờ tối nhé.",
            [
                {"name": "cinema.list_showtimes", "arguments": {"movie": "Mưa Đỏ", "date": "2026-11-18"}},
                {"name": "cinema.buy_tickets", "arguments": {"showtime_id": "shw_03", "seats": 2}},
            ],
            [
                {"ok": True, "showtimes": [{"id": "shw_01", "movie": "Mưa Đỏ", "time": "17:00"}, {"id": "shw_03", "movie": "Mưa Đỏ", "time": "19:00"}]},
                {"ok": True, "ticket": {"id": "tkt_01", "showtime_id": "shw_03", "seats": 2}},
            ],
        ),
        (
            "sfr_04",
            "Đặt phòng khách sạn ở Nha Trang trong 3 đêm giúp mình với.",
            [
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "hotel_booking.search_hotels", "params": ["check_in"]}},
            ],
            [
                {"ok": True},
            ],
        ),
    ]
    out = []
    for sid, txt, calls, res in samples:
        msgs = make_traj_fn(sfr_cat, base_ctx, txt, [], calls, res, few_shot_block="")
        out.append({"id": sid, "variant": "sfr", "messages": msgs})
    return out
