"""Chuẩn hóa transcript ASR:
1. Unicode NFC + làm sạch khoảng trắng.
2. Khôi phục từ viết tắt / tên kênh / tên ứng dụng bị ASR phiên âm rời (VOV, VTV, HTV, SJC, SMS, RON95, Zalo, S+ Sport...).
3. Đổi dãy chữ số đọc liên tiếp (>= 7 chữ số, ví dụ số điện thoại / số tài khoản) thành chuỗi số, kể cả khi ASR nghe nhầm nhẹ ('chin' -> '9').
"""
from __future__ import annotations

import re
import unicodedata

_DIGITS = {
    "không": "0",
    "một": "1", "mốt": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5", "nhăm": "5",
    "sáu": "6",
    "bảy": "7", "bẩy": "7",
    "tám": "8",
    "chín": "9", "chin": "9", "chính": "9",
}

# Các mẫu phiên âm rời phổ biến của PhoWhisper/Whisper sang dạng chuẩn trong bộ dữ liệu
_PHONETIC_FIXES = [
    # Đài / Kênh truyền hình & phát thanh
    (r"\b(?:về|vê)\s*ô\s*vê\s*giao\s*thông\b", "VOV Giao thông"),
    (r"\b(?:về|vê)\s*ô\s*vê\s*(?:một|1)\b", "VOV1"),
    (r"\b(?:về|vê)\s*ô\s*vê\s*(?:hai|2)\b", "VOV2"),
    (r"\b(?:về|vê)\s*ô\s*vê\s*(?:ba|bá|3)\b", "VOV3"),
    (r"\b(?:về|vê)\s*ô\s*vê\b", "VOV"),
    (r"\b(?:vê\s*tê\s*vê|vtv)\s*(?:một|1)\b", "VTV1"),
    (r"\b(?:vê\s*tê\s*vê|vtv)\s*(?:hai|2)\b", "VTV2"),
    (r"\b(?:vê\s*tê\s*vê|vtv)\s*(?:ba|3)\b", "VTV3"),
    (r"\b(?:vê\s*tê\s*vê|vtv)\s*(?:sáu|6)\b", "VTV6"),
    (r"\b(?:hát\s*(?:ti\s*vi|tê\s*vê)|htv)\s*(?:bảy|bẩy|7)\b", "HTV7"),
    (r"\b(?:hát\s*(?:ti\s*vi|tê\s*vê)|htv)\s*(?:chín|9)\b", "HTV9"),
    (r"\b(?:ét\s*cộng|s\s*cộng|s\+|s\s*com)\s*sports?\b", "S+ Sport"),
    (r"\bsctv\s*(?:chín|9)\b", "SCTV9"),
    (r"\bti\s+vi\b", "tivi"),
    # Tài chính / Xăng dầu / Viễn thông / Ứng dụng / Chứng khoán / Tàu xe
    (r"\bvàng\s+(?:s\s*c\s*c|sdc|sjc|ét\s*(?:dây|giây|di)\s*xi)\b", "vàng SJC"),
    (r"\b(?:xăng|sang)\s+(?:ron|rôn|roi|giàn|gian)\s*(?:chín\s*(?:mươi\s*)?lăm|chính\s*làm|chỉ\s*đang|chiến\s*lâm|95)\b", "xăng RON95"),
    (r"\b(?:xăng|sang)\s+e\s*(?:năm|5)\b", "xăng E5"),
    (r"\b(?:ét\s*em\s*(?:ét|ế)|s\s*m\s*s)\b", "SMS"),
    (r"\b(?:giao|gia|da)\s+lô\b", "Zalo"),
    (r"\b(?:vê|vi)\s*(?:en|n)\s*(?:in\s*(?:đếch|đéc|dex)|index)\b", "VNINDEX"),
    (r"\bvn\s*[-–]?\s*index\b", "VNINDEX"),
    (r"\b(?:vê|vi)\s*(?:en|n)\s*(?:ba\s*mươi|30)\b", "VN30"),
    (r"\b(?:dét|dép|zết|giết|z)\s*tê\s*pê\b", "ZTP"),
    (r"\b(?:dét|dép|zết|giết|z)\s*mờ\s*(?:ích|x)\b", "ZMX"),
    (r"\b(?:dét|dép|zết|giết|z)\s*đê\s*hát\b", "ZDH"),
    (r"\b(?:dét|dép|zết|giết|z)\s*đê\s*vê\b", "ZDV"),
    (r"\b(?:dét|dép|zết|giết|z)\s*(?:ca|k)\s*(?:en|n)\b", "ZKN"),
    (r"\b(?:dét|dép|zết|giết|z)\s*tê\s*(?:ét|s)\b", "ZTS"),
    (r"\b(?:dét|dép|zết|giết|z)\s*bê\s*hát\b", "ZBH"),
    (r"\b(?:dét|dép|zết|giết|z)\s*vê\s*tê\b", "ZVT"),
    (r"\bvi\s*tếch\b", "Vitech"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:một|1)\b", "tàu SE1"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:hai|2)\b", "tàu SE2"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:ba|3)\b", "tàu SE3"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:bốn|4)\b", "tàu SE4"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:năm|5)\b", "tàu SE5"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:sáu|6)\b", "tàu SE6"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:bảy|bẩy|7)\b", "tàu SE7"),
    (r"\btàu\s+(?:ét\s*e|se)\s*(?:tám|8)\b", "tàu SE8"),
    (r"\b(?:thép|khép)\s+(?:đại|lại)\s+(?:hưng|hương)\b", "Thép Đại Hưng"),
    (r"\bjack\s*kanner\b", "ZKN"),
    (r"\bzalo\s*nhỏ\b", "ZKN"),
    (r"\bcompact\s+ca\s+(?:hát|hai|2)\b", "Compact K2"),
    (r"\btàu\s+s\s*&?\s*p\s*(?:ba|3)\b", "tàu SE3"),
    (r"\bsân\s+bay\s+nửa\s+(?:bài|bãi)\b", "sân bay Nội Bài"),
    (r"\bchuyến\s+r\s*v\s*(\d+)\b", r"chuyến VJ\1"),
    (r"\bliên\s+hoan\s+(?:xin|sim)\s+nhắn\b", "Liên hoan phim ngắn"),
    (r"\b(?:sắp|súp)\s+bí\s+đỏ\b", "súp bí đỏ"),
    (r"\bmình\s+xào\s+chay\b", "mì xào chay"),
    (r"\bin\s+mặt\b", "yến mạch"),
    (r"\bben\s+kích\b", "pancake"),
]
_COMPILED_FIXES = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _PHONETIC_FIXES]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean(text: str) -> str:
    """NFC + gộp khoảng trắng + strip."""
    return re.sub(r"\s+", " ", nfc(text)).strip()


def fix_phonetic_terms(text: str) -> str:
    """Khôi phục các từ viết tắt/tên riêng bị ASR phiên âm rời."""
    for rx, rep in _COMPILED_FIXES:
        text = rx.sub(rep, text)
    return text


def digit_words_to_number(text: str, min_len: int = 7) -> str:
    """Đổi dãy >= min_len từ chữ số/cụm số liên tiếp thành chuỗi số (số điện thoại, số tài khoản).

    Hỗ trợ cả trường hợp ASR đã đổi một phần thành số ('090123499 chin sáu' -> '09012349996').
    """
    tokens = text.split(" ")
    out, run, run_digits = [], [], []

    def flush():
        total_digits = "".join(run_digits)
        if len(total_digits) >= min_len:
            out.append(total_digits)
        else:
            out.extend(run)
        run.clear()
        run_digits.clear()

    for tok in tokens:
        bare = tok.strip(".,;!?").lower()
        if bare in _DIGITS:
            run.append(tok)
            run_digits.append(_DIGITS[bare])
        elif bare.isdigit() and (run_digits or len(bare) >= 3):
            run.append(tok)
            run_digits.append(bare)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def postprocess(text: str, digit_sequences: bool = True, phonetic_fixes: bool = True) -> str:
    text = clean(text)
    if phonetic_fixes:
        text = fix_phonetic_terms(text)
    if digit_sequences:
        text = digit_words_to_number(text)
    return text
