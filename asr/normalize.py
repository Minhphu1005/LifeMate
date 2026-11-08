"""Chuẩn hóa transcript: NFC, strip, gộp khoảng trắng, đổi dãy chữ số đọc thành số."""
from __future__ import annotations

import re
import unicodedata

_DIGITS = {
    "không": "0",
    "một": "1", "mốt": "1",
    "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5", "sáu": "6", "bảy": "7", "bẩy": "7",
    "tám": "8", "chín": "9",
}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean(text: str) -> str:
    """NFC + gộp khoảng trắng + strip."""
    return re.sub(r"\s+", " ", nfc(text)).strip()


def digit_words_to_number(text: str, min_len: int = 7) -> str:
    """Đổi dãy >= min_len từ chữ số liên tiếp ("không chín năm ...") thành số điện thoại.

    Chỉ áp dụng cho dãy dài (số điện thoại, số tài khoản) để tránh đổi nhầm
    những câu như "hai ba người". Các số đọc theo hàng ("năm mươi nghìn") để
    nguyên cho LLM xử lý.
    """
    tokens = text.split(" ")
    out, run = [], []

    def flush():
        if len(run) >= min_len:
            out.append("".join(_DIGITS[t.strip(".,;!?").lower()] for t in run))
        else:
            out.extend(run)
        run.clear()

    for tok in tokens:
        bare = tok.strip(".,;!?").lower()
        if bare in _DIGITS:
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


def postprocess(text: str, digit_sequences: bool = True) -> str:
    text = clean(text)
    if digit_sequences:
        text = digit_words_to_number(text)
    return text
