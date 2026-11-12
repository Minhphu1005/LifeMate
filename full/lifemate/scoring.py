"""Chấm điểm offline đúng công thức của đề (overview.md, mục Evaluation)."""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Set


def _toks(s) -> List[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(a, b) -> float:
    ta, tb = _toks(a), _toks(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def call_match(sub: dict, gold: dict, fuzzy: Dict[str, Set[str]]) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    fz = fuzzy.get(gold["name"], set())
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    for k in set(sa) | set(ga):
        if k in fz:
            if token_f1(sa.get(k, ""), ga.get(k, "")) <= 0.6:
                return False
        elif sa.get(k) != ga.get(k):
            return False
    return True


def max_matches(sub: List[dict], gold: List[dict], fuzzy) -> int:
    """Ghép cặp tối đa (mỗi call chỉ khớp một call) bằng đường tăng luồng."""
    adj = [[j for j, g in enumerate(gold) if call_match(s, g, fuzzy)] for s in sub]
    owner = [-1] * len(gold)

    def try_(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] < 0 or try_(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(1 for i in range(len(sub)) if try_(i, set()))


def score_sample(sub_calls: List[dict], gold_calls: List[dict], hash_ok: bool, fuzzy) -> dict:
    if not gold_calls:
        s = 1.0 if not sub_calls else 0.0
        return {"score": s, "M": 0, "U": len(sub_calls), "G": 0, "hash_ok": hash_ok}
    M = max_matches(sub_calls, gold_calls, fuzzy)
    U = len(sub_calls) - M
    if not hash_ok or M == 0:
        return {"score": 0.0, "M": M, "U": U, "G": len(gold_calls), "hash_ok": hash_ok}
    p, r = M / (M + U), M / len(gold_calls)
    return {"score": 2 * p * r / (p + r), "M": M, "U": U, "G": len(gold_calls), "hash_ok": hash_ok}

