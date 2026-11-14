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

_PHONETIC_FIXES = [
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
    (r"\bvàng\s+(?:s\s*c\s*c|sdc|sjc|ét\s*(?:dây|giây|di)\s*xi)\b", "vàng SJC"),
    (r"\b(?:xăng|sang)\s+(?:ron|rôn|roi|giàn|gian)\s*(?:chín\s*(?:mươi\s*)?lăm|chính\s*làm|chỉ\s*đang|chiến\s*lâm|95)\b", "xăng RON95"),
    (r"\b(?:xăng|sang)\s+e\s*(?:năm|5)\b", "xăng E5"),
    (r"\b(?:ét\s*em\s*(?:ét|ế)|s\s*m\s*s)\b", "SMS"),
    (r"\b(?:giao|gia|da)\s+lô\b", "Zalo"),
]
_COMPILED_FIXES = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in _PHONETIC_FIXES]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", nfc(text)).strip()


def fix_phonetic_terms(text: str) -> str:
    for rx, rep in _COMPILED_FIXES:
        text = rx.sub(rep, text)
    return text


def digit_words_to_number(text: str, min_len: int = 7) -> str:
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
