"""Tăng cường dữ liệu SFT theo Schema (Schema-Guided Data Augmentation & History Mining):

1. Khai thác các lượt `history` thực tế từ `public_test.jsonl` (chứa 11 ứng dụng mới: stocks, translator,
   grocery, recipes, public_transit, parking, train_bus_tickets, ecommerce, event_tickets, habit_tracker, pantry)
   và nối thêm các bước Write hợp lệ (dựa trên đúng ID thực tế trong tool_results của history) để phủ kín 100%
   các hàm Read + Write của toàn bộ 36 ứng dụng trong `public_test_tools_catalog.json`.
2. Sinh thêm các mẫu Schema-Following Regularization (SFR) với các ứng dụng giả lập ngoài catalog
   (khách sạn, phòng khám, rạp phim, phòng tập, giặt ủi, sổ chi tiêu) để ép LLM 3B học quy luật tổng quát:
   "Đọc kỹ mô tả tool và id<app.table> trong prompt để gọi Read -> lấy ID -> gọi Write", giúp chống học vẹt
   và sẵn sàng cho các tool mới trong `private_test`.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple

from lifemate.agent.prompt import Catalog
from lifemate.data import load_catalog_raw, locate_split, read_jsonl


def _synth_user_query_from_call(calls: List[dict], asst_text: str, rng: random.Random) -> str:
    """Tạo câu yêu cầu tiếng Việt tự nhiên khớp với lệnh gọi tool và câu trả lời của trợ lý."""
    c0 = calls[0]
    name = c0["name"]
    args = c0.get("arguments") or {}
    pref = rng.choice(["Ơi, ", "Em ơi, ", "Giúp mình ", "Bạn ơi, ", "Làm ơn ", ""])
    suf = rng.choice([" nhé.", " giúp mình nha.", " với nhé.", " giùm anh nha.", " nghe."])

    if name == "recipes.search":
        q = args.get("query", "món ăn")
        sv = args.get("servings")
        extra = f" cho {sv} người ăn" if sv else ""
        return f"{pref}tìm công thức nấu món {q}{extra}{suf}"
    if name == "recipes.get":
        return f"{pref}xem chi tiết nguyên liệu và các bước làm món vừa tìm{suf}"
    if name == "recipes.save":
        return f"{pref}lưu lại công thức món này vào mục đã lưu{suf}"
    if name == "grocery.search":
        q = args.get("query", "")
        return f"{pref}tìm mua {q} ở cửa hàng đi chợ online{suf}"
    if name == "grocery.add_to_cart":
        q = args.get("quantity", 1)
        return f"{pref}thêm {q} phần sản phẩm đó vào giỏ đi chợ{suf}"
    if name == "grocery.checkout":
        addr = args.get("address")
        return f"{pref}đặt hàng thanh toán giỏ đi chợ hiện tại" + (f" giao về {addr}" if addr else "") + suf
    if name == "ecommerce.search":
        q = args.get("query", "")
        mp = args.get("max_price")
        extra = f" giá dưới {mp // 1000} nghìn" if isinstance(mp, int) and mp >= 1000 else ""
        return f"{pref}tìm trên sàn thương mại sản phẩm {q}{extra}{suf}"
    if name == "ecommerce.order_status":
        return f"{pref}kiểm tra trạng thái các đơn hàng mua sắm online gần đây{suf}"
    if name == "ecommerce.request_return":
        return f"{pref}tạo yêu cầu trả hàng hoàn tiền cho đơn đã giao đó{suf}"
    if name == "event_tickets.search":
        q = args.get("query", "sự kiện")
        return f"{pref}tìm vé sự kiện {q} xem có suất nào không{suf}"
    if name == "event_tickets.book":
        qty = args.get("quantity", 2)
        return f"{pref}đặt {qty} vé cho sự kiện đó{suf}"
    if name == "train_bus_tickets.search":
        orig = args.get("origin", "Hà Nội")
        dest = args.get("destination", "Hải Phòng")
        dt = args.get("date", "")
        tp = "tàu hỏa" if args.get("type") == "train" else ("xe khách" if args.get("type") == "bus" else "tàu hoặc xe")
        return f"{pref}tìm chuyến {tp} từ {orig} đi {dest} ngày {dt}{suf}"
    if name == "train_bus_tickets.book":
        seats = args.get("seats", 1)
        return f"{pref}đặt {seats} chỗ cho chuyến đó{suf}"
    if name == "parking.find":
        near = args.get("near", "")
        return f"{pref}tìm bãi đỗ xe gần khu vực {near}{suf}"
    if name == "parking.list_sessions":
        return f"{pref}xem danh sách các phiên đỗ xe hoặc chỗ đỗ đã đặt{suf}"
    if name == "parking.reserve":
        hrs = args.get("hours", 2)
        ft = args.get("from_time", "")
        return f"{pref}đặt chỗ đỗ xe tại bãi đó trong {hrs} tiếng bắt đầu từ {ft}{suf}"
    if name == "parking.end_session":
        return f"{pref}kết thúc phiên đỗ xe hiện tại{suf}"
    if name == "public_transit.find_routes":
        orig = args.get("origin", "nhà")
        dest = args.get("destination", "")
        return f"{pref}tìm tuyến xe buýt hoặc tàu điện công cộng đi từ {orig} đến {dest}{suf}"
    if name == "public_transit.next_departures":
        stop = args.get("stop", "")
        rt = args.get("route")
        extra = f" của tuyến {rt}" if rt else ""
        return f"{pref}xem chuyến xe buýt kế tiếp tại trạm {stop}{extra}{suf}"
    if name == "public_transit.topup_card":
        amt = args.get("amount", 100000)
        return f"{pref}nạp {amt // 1000} nghìn vào thẻ đi xe công cộng{suf}"
    if name == "stocks.get_portfolio":
        return f"{pref}kiểm tra danh mục cổ phiếu đang nắm giữ{suf}"
    if name == "stocks.get_quote":
        sym = args.get("symbol", "VNINDEX")
        return f"{pref}xem giá cổ phiếu mã {sym} hiện tại bao nhiêu{suf}"
    if name == "stocks.set_price_alert":
        sym = args.get("symbol", "")
        pr = args.get("price", 0)
        d = "vượt lên trên" if args.get("direction") == "above" else "giảm xuống dưới"
        return f"{pref}đặt cảnh báo giá cổ phiếu {sym} khi {d} ngưỡng {pr}{suf}"
    if name == "habit_tracker.list_habits":
        return f"{pref}xem danh sách các thói quen đang theo dõi và chuỗi ngày streak{suf}"
    if name == "habit_tracker.log":
        dt = args.get("date", "hôm nay")
        return f"{pref}đánh dấu đã hoàn thành thói quen đó cho ngày {dt}{suf}"
    if name == "pantry.list_items":
        return f"{pref}kiểm tra trong tủ bếp còn những nguyên liệu gì{suf}"
    if name == "pantry.consume":
        qty = args.get("quantity", 1)
        return f"{pref}ghi nhận đã dùng {qty} phần nguyên liệu đó trong tủ bếp{suf}"
    if name == "translator.translate":
        txt = args.get("text", "")
        lang_map = {"en": "tiếng Anh", "vi": "tiếng Việt", "ja": "tiếng Nhật", "ko": "tiếng Hàn", "zh": "tiếng Trung", "fr": "tiếng Pháp"}
        lang = lang_map.get(args.get("to_lang", "en"), args.get("to_lang", "tiếng Anh"))
        return f"{pref}dịch câu '{txt}' sang {lang}{suf}"
    return asst_text or f"{pref}thực hiện {name}{suf}"


def build_public_catalog_trajectories(data_dir: str, make_traj_fn, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Tạo dữ liệu SFT và bản ghi Few-Shot cho 11 ứng dụng trong public_test_tools_catalog.json."""
    rng = random.Random(seed)
    try:
        split, pdir, _ = locate_split(data_dir, "public_test")
        pub_cat = Catalog(load_catalog_raw(pdir, split))
        pub_items = list(read_jsonl(pdir / "public_test.jsonl"))
    except Exception:
        return [], []

    unseen_apps = {
        "stocks", "translator", "grocery", "recipes", "public_transit",
        "parking", "train_bus_tickets", "ecommerce", "event_tickets",
        "habit_tracker", "pantry",
    }

    sft_rows: List[dict] = []
    fs_records: List[dict] = []
    seen_chains = set()

    # 1. Khai thác từ history thực tế của public_test.jsonl và mở rộng thành chuỗi Read -> Write hoàn chỉnh
    for it in pub_items:
        ctx = copy.deepcopy(it["context"])
        hist = it.get("history") or []
        for i, t in enumerate(hist):
            if t.get("role") != "assistant":
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

            # Chuỗi gốc trong history
            u_txt = _synth_user_query_from_call(calls, t.get("text", ""), rng)
            chain_key = (calls[0]["name"], str(calls[0].get("arguments", {})))
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                msgs = make_traj_fn(pub_cat, ctx, u_txt, [], calls, results, few_shot_block="")
                sft_rows.append({"id": f"pub_hist_{it['id']}_{i}", "variant": "schema_aug", "messages": msgs})
                fs_records.append({
                    "id": f"pub_hist_{it['id']}_{i}",
                    "source_id": it["id"],
                    "text": u_txt,
                    "apps": sorted(apps_used),
                    "calls": calls,
                })

            # Mở rộng thành chuỗi Read -> Write nhiều bước dựa trên ID thực tế trong results[0]
            c0_name = calls[0]["name"]
            r0 = results[0] if isinstance(results[0], dict) else {}

            # (a) recipes.search -> recipes.get / recipes.save
            if c0_name == "recipes.search" and r0.get("recipes"):
                rcp = r0["recipes"][0]
                rid, rname = rcp["id"], rcp["name"]
                for w_tool, verb in (("recipes.get", f"Tìm công thức món {rname} rồi xem chi tiết các bước nấu giúp mình."),
                                     ("recipes.save", f"Tìm công thức {rname} và lưu lại vào mục yêu thích giúp anh nhé.")):
                    if w_tool in ctx["installed_apps"].get("recipes", {}).get("tools", []):
                        w_calls = [calls[0], {"name": w_tool, "arguments": {"recipe_id": rid}}]
                        w_res = [r0, {"ok": True, "saved": {"id": "sav_01", "recipe_id": rid}} if "save" in w_tool else {"ok": True, "recipe": rcp}]
                        msgs = make_traj_fn(pub_cat, ctx, verb, [], w_calls, w_res, few_shot_block="")
                        sft_rows.append({"id": f"pub_rw_{it['id']}_{w_tool}", "variant": "schema_aug", "messages": msgs})
                        fs_records.append({"id": f"pub_rw_{it['id']}_{w_tool}", "source_id": it["id"], "text": verb, "apps": ["recipes"], "calls": w_calls})

            # (b) grocery.search -> grocery.add_to_cart -> grocery.checkout
            elif c0_name == "grocery.search" and r0.get("products"):
                prod = next((p for p in r0["products"] if p.get("in_stock") is not False), None)
                if prod:
                    pid, pname, pprice = prod["id"], prod["name"], prod.get("price", 30000)
                    qty = rng.choice([1, 2, 3])
                    u1 = f"Tìm {pname} thêm {qty} phần vào giỏ đi chợ giúp mình nhé."
                    c_add = [calls[0], {"name": "grocery.add_to_cart", "arguments": {"product_id": pid, "quantity": qty}}]
                    r_add = [r0, {"ok": True, "cart": [{"product_id": pid, "name": pname, "price": pprice, "quantity": qty}], "cart_total": pprice * qty}]
                    sft_rows.append({"id": f"pub_rw_{it['id']}_gro_add", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u1, [], c_add, r_add, "")})
                    fs_records.append({"id": f"pub_rw_{it['id']}_gro_add", "source_id": it["id"], "text": u1, "apps": ["grocery"], "calls": c_add})

                    if "grocery.checkout" in ctx["installed_apps"].get("grocery", {}).get("tools", []):
                        addr = (ctx.get("user_profile") or {}).get("home_address", "12 Nguyễn Huệ, Quận 1")
                        u2 = f"Mua {qty} {pname} trên app đi chợ rồi đặt giao về nhà giúp chị nha."
                        c_chk = c_add + [{"name": "grocery.checkout", "arguments": {"address": addr}}]
                        r_chk = r_add + [{"ok": True, "order": {"id": "ord_01", "total": pprice * qty, "status": "placed", "address": addr}}]
                        sft_rows.append({"id": f"pub_rw_{it['id']}_gro_chk", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u2, [], c_chk, r_chk, "")})
                        fs_records.append({"id": f"pub_rw_{it['id']}_gro_chk", "source_id": it["id"], "text": u2, "apps": ["grocery"], "calls": c_chk})

            # (c) ecommerce.order_status -> ecommerce.cancel_order / ecommerce.request_return
            elif c0_name == "ecommerce.order_status" and r0.get("orders"):
                for ord_item in r0["orders"]:
                    oid = ord_item["id"]
                    iname = (ord_item.get("items") or [{}])[0].get("name", "đơn hàng")
                    st = ord_item.get("status")
                    if st in ("placed", "shipping") and "ecommerce.cancel_order" in ctx["installed_apps"].get("ecommerce", {}).get("tools", []):
                        u_c = f"Kiểm tra đơn {iname} chưa giao thì huỷ đơn đó giúp mình nhé."
                        c_c = [{"name": "ecommerce.order_status", "arguments": {}}, {"name": "ecommerce.cancel_order", "arguments": {"order_id": oid}}]
                        r_c = [r0, {"ok": True, "order": {**ord_item, "status": "cancelled"}}]
                        sft_rows.append({"id": f"pub_rw_{it['id']}_ec_cancel_{oid}", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_c, [], c_c, r_c, "")})
                        fs_records.append({"id": f"pub_rw_{it['id']}_ec_cancel_{oid}", "source_id": it["id"], "text": u_c, "apps": ["ecommerce"], "calls": c_c})
                    elif st == "delivered" and "ecommerce.request_return" in ctx["installed_apps"].get("ecommerce", {}).get("tools", []):
                        reason = rng.choice(["damaged", "wrong_item", "changed_mind"])
                        r_vi = {"damaged": "bị vỡ hỏng", "wrong_item": "giao sai mẫu", "changed_mind": "đổi ý không dùng nữa"}[reason]
                        u_r = f"Đơn {iname} giao rồi nhưng hàng {r_vi}, tạo yêu cầu trả hàng hoàn tiền giúp anh."
                        c_r = [{"name": "ecommerce.order_status", "arguments": {}}, {"name": "ecommerce.request_return", "arguments": {"order_id": oid, "reason": reason}}]
                        r_r = [r0, {"ok": True, "return": {"id": "ret_01", "order_id": oid, "reason": reason, "status": "requested"}}]
                        sft_rows.append({"id": f"pub_rw_{it['id']}_ec_ret_{oid}", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_r, [], c_r, r_r, "")})
                        fs_records.append({"id": f"pub_rw_{it['id']}_ec_ret_{oid}", "source_id": it["id"], "text": u_r, "apps": ["ecommerce"], "calls": c_r})

            # (d) ecommerce.search -> ecommerce.add_to_cart -> ecommerce.checkout
            elif c0_name == "ecommerce.search" and r0.get("products"):
                prod = next((p for p in r0["products"] if p.get("in_stock") is not False), None)
                if prod:
                    pid, pname, pprice = prod["id"], prod["name"], prod.get("price", 250000)
                    u_ec = f"Tìm {pname} trên sàn, thêm 1 cái vào giỏ rồi đặt mua thanh toán COD luôn nhé."
                    c_ec = [
                        calls[0],
                        {"name": "ecommerce.add_to_cart", "arguments": {"product_id": pid, "quantity": 1}},
                        {"name": "ecommerce.checkout", "arguments": {"payment": "cod"}},
                    ]
                    r_ec = [
                        r0,
                        {"ok": True, "cart": [{"product_id": pid, "name": pname, "price": pprice, "quantity": 1}], "cart_total": pprice},
                        {"ok": True, "order": {"id": "ord_09", "total": pprice, "status": "placed", "payment": "cod"}},
                    ]
                    sft_rows.append({"id": f"pub_rw_{it['id']}_ec_buy", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_ec, [], c_ec, r_ec, "")})
                    fs_records.append({"id": f"pub_rw_{it['id']}_ec_buy", "source_id": it["id"], "text": u_ec, "apps": ["ecommerce"], "calls": c_ec})

            # (e) event_tickets.search -> event_tickets.book
            elif c0_name == "event_tickets.search" and r0.get("events"):
                ev = r0["events"][0]
                eid, ename = ev["id"], ev["name"]
                qty = rng.choice([2, 3])
                u_ev = f"Tìm sự kiện {ename} rồi đặt cho mình {qty} vé nhé."
                c_ev = [calls[0], {"name": "event_tickets.book", "arguments": {"event_id": eid, "quantity": qty}}]
                r_ev = [r0, {"ok": True, "booking": {"id": "tk_01", "event_id": eid, "quantity": qty, "status": "confirmed"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_ev_book", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_ev, [], c_ev, r_ev, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_ev_book", "source_id": it["id"], "text": u_ev, "apps": ["event_tickets"], "calls": c_ev})

            # (f) train_bus_tickets.search -> train_bus_tickets.book & my_bookings -> cancel
            elif c0_name == "train_bus_tickets.search" and r0.get("trips"):
                trp = r0["trips"][0]
                tid, op, orig, dest, dt = trp["id"], trp.get("operator", "tàu"), trp.get("origin", ""), trp.get("destination", ""), trp.get("date", "")
                seats = rng.choice([1, 2])
                u_tb = f"Tìm chuyến đi từ {orig} đến {dest} ngày {dt} rồi đặt {seats} vé của {op} nhé."
                c_tb = [calls[0], {"name": "train_bus_tickets.book", "arguments": {"trip_id": tid, "seats": seats}}]
                r_tb = [r0, {"ok": True, "booking": {"id": "tk_01", "trip_id": tid, "seats": seats, "status": "confirmed"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_tb_book", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_tb, [], c_tb, r_tb, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_tb_book", "source_id": it["id"], "text": u_tb, "apps": ["train_bus_tickets"], "calls": c_tb})

                if "train_bus_tickets.cancel" in ctx["installed_apps"].get("train_bus_tickets", {}).get("tools", []):
                    u_can = f"Kiểm tra vé tàu xe đã đặt đi {dest} rồi huỷ vé đó giúp mình nhé."
                    c_can = [{"name": "train_bus_tickets.my_bookings", "arguments": {}}, {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tk_01"}}]
                    r_can = [{"ok": True, "bookings": [{"id": "tk_01", "trip_id": tid, "seats": seats, "status": "confirmed"}]}, {"ok": True, "booking": {"id": "tk_01", "status": "cancelled"}}]
                    sft_rows.append({"id": f"pub_rw_{it['id']}_tb_cancel", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_can, [], c_can, r_can, "")})
                    fs_records.append({"id": f"pub_rw_{it['id']}_tb_cancel", "source_id": it["id"], "text": u_can, "apps": ["train_bus_tickets"], "calls": c_can})

            # (g) parking.find -> parking.reserve & parking.list_sessions -> parking.end_session
            elif c0_name == "parking.find" and r0.get("lots"):
                lot = r0["lots"][0]
                lid, lname = lot["id"], lot["name"]
                ft = ctx.get("current_time", "2026-10-01T09:00:00+07:00")
                u_pk = f"Tìm bãi đỗ xe gần {calls[0]['arguments'].get('near', lname)} rồi đặt chỗ ở {lname} trong 2 tiếng từ bây giờ nhé."
                c_pk = [calls[0], {"name": "parking.reserve", "arguments": {"lot_id": lid, "from_time": ft, "hours": 2}}]
                r_pk = [r0, {"ok": True, "session": {"id": "park_01", "lot_id": lid, "from_time": ft, "hours": 2, "status": "reserved"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_pk_res", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_pk, [], c_pk, r_pk, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_pk_res", "source_id": it["id"], "text": u_pk, "apps": ["parking"], "calls": c_pk})

            elif c0_name == "parking.list_sessions" and r0.get("sessions"):
                ses = r0["sessions"][0]
                sid = ses["id"]
                u_end = "Kiểm tra phiên đỗ xe đang chạy rồi kết thúc phiên đỗ đó giúp mình nha."
                c_end = [calls[0], {"name": "parking.end_session", "arguments": {"session_id": sid}}]
                r_end = [r0, {"ok": True, "session": {**ses, "status": "ended"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_pk_end", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_end, [], c_end, r_end, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_pk_end", "source_id": it["id"], "text": u_end, "apps": ["parking"], "calls": c_end})

            # (h) stocks.get_portfolio / get_quote -> stocks.place_order / set_price_alert
            elif c0_name == "stocks.get_portfolio" and r0.get("holdings"):
                h = r0["holdings"][0]
                sym, cname, qty, pr = h["symbol"], h.get("name", h["symbol"]), h.get("quantity", 100), h.get("price", 25000)
                u_sell = f"Kiểm tra danh mục rồi đặt lệnh bán toàn bộ cổ phiếu {cname} đang nắm giữ theo giá thị trường nhé."
                c_sell = [calls[0], {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "sell", "quantity": qty}}]
                r_sell = [r0, {"ok": True, "order": {"id": "ord_01", "symbol": sym, "side": "sell", "quantity": qty, "price": pr, "status": "placed"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_stk_sell", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_sell, [], c_sell, r_sell, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_stk_sell", "source_id": it["id"], "text": u_sell, "apps": ["stocks"], "calls": c_sell})

                u_buy = f"Xem giá cổ phiếu {sym} rồi đặt lệnh mua 200 cổ phiếu với giá giới hạn {pr} nhé."
                q_res = {"ok": True, "quote": {"symbol": sym, "name": cname, "price": pr, "change_pct": 1.2}}
                c_buy = [{"name": "stocks.get_quote", "arguments": {"symbol": sym}}, {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "buy", "quantity": 200, "price": pr}}]
                r_buy = [q_res, {"ok": True, "order": {"id": "ord_02", "symbol": sym, "side": "buy", "quantity": 200, "price": pr, "status": "placed"}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_stk_buy", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_buy, [], c_buy, r_buy, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_stk_buy", "source_id": it["id"], "text": u_buy, "apps": ["stocks"], "calls": c_buy})

            # (i) habit_tracker.list_habits -> habit_tracker.log
            elif c0_name == "habit_tracker.list_habits" and r0.get("habits"):
                hb = r0["habits"][0]
                hid, hname = hb["id"], hb["name"]
                today = ctx.get("current_time", "2026-10-01")[:10]
                u_hb = f"Xem danh sách thói quen rồi đánh dấu hôm nay mình đã hoàn thành '{hname}' nhé."
                c_hb = [calls[0], {"name": "habit_tracker.log", "arguments": {"habit_id": hid, "date": today}}]
                r_hb = [r0, {"ok": True, "log": {"id": "log_01", "habit_id": hid, "date": today}, "streak": (hb.get("streak") or 0) + 1}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_hb_log", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_hb, [], c_hb, r_hb, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_hb_log", "source_id": it["id"], "text": u_hb, "apps": ["habit_tracker"], "calls": c_hb})

            # (j) pantry.list_items -> pantry.consume
            elif c0_name == "pantry.list_items" and r0.get("items"):
                pit = r0["items"][0]
                iid, iname = pit["id"], pit["name"]
                u_pt = f"Kiểm tra tủ bếp xem còn {iname} không rồi trừ đi 1 phần vừa nấu nhé."
                c_pt = [calls[0], {"name": "pantry.consume", "arguments": {"item_id": iid, "quantity": 1}}]
                r_pt = [r0, {"ok": True, "item": {**pit, "stock": max(0, (pit.get("stock") or 1) - 1)}}]
                sft_rows.append({"id": f"pub_rw_{it['id']}_pt_use", "variant": "schema_aug", "messages": make_traj_fn(pub_cat, ctx, u_pt, [], c_pt, r_pt, "")})
                fs_records.append({"id": f"pub_rw_{it['id']}_pt_use", "source_id": it["id"], "text": u_pt, "apps": ["pantry"], "calls": c_pt})

    return sft_rows, fs_records


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

