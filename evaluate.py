"""Chấm điểm offline theo công thức của đề (F1 trên tool_calls, 0 nếu hash sai hoặc M=0).

  python evaluate.py --pred submission.jsonl --data-dir <ROOT> --split train
  python evaluate.py --self-test --data-dir <ROOT> --split train     # nộp = gold -> 100 điểm
Tuỳ chọn --gold-hash gold_hash.jsonl ({"id","state_hash"}) để chấm cả hash; không có thì
chỉ chấm tool_calls (báo rõ "hash_checked": false).
Gold lấy từ label.tool_calls trong <split>.jsonl (chỉ có ở train).
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.catalog import fuzzy_params, load_catalog, resolve_layout


def _tokens(s) -> List[str]:
    return unicodedata.normalize("NFC", str(s)).lower().split()


def token_f1(a, b) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def calls_match(sub: dict, gold: dict, fuzzy: Dict[str, set]) -> bool:
    if sub.get("name") != gold.get("name"):
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    fz = fuzzy.get(gold["name"], set())
    if set(sa) != set(ga):
        return False
    for k, gv in ga.items():
        sv = sa[k]
        if k in fz:
            if token_f1(gv, sv) <= 0.6:
                return False
        elif sv != gv:
            return False
    return True


def count_matches(sub: List[dict], gold: List[dict], fuzzy) -> int:
    """Ghép 1-1 tối đa (mỗi call khớp tối đa 1 gold); không xét thứ tự."""
    adj = [[j for j, g in enumerate(gold) if calls_match(s, g, fuzzy)] for s in sub]
    owner = [-1] * len(gold)

    def try_(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or try_(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(try_(i, set()) for i in range(len(sub)))


def score_sample(sub: List[dict], gold: List[dict], fuzzy, hash_ok: Optional[bool]) -> dict:
    if not gold:
        s = 1.0 if not sub else 0.0
        return {"score": s, "M": 0, "U": len(sub), "P": 0, "R": 0}
    m = count_matches(sub, gold, fuzzy)
    u = len(sub) - m
    if m == 0 or hash_ok is False:
        return {"score": 0.0, "M": m, "U": u, "P": 0, "R": 0}
    p, r = m / (m + u), m / len(gold)
    return {"score": 2 * p * r / (p + r), "M": m, "U": u, "P": p, "R": r}


def evaluate(pred: Dict[str, dict], rows: List[dict], fuzzy, gold_hash=None) -> dict:
    per, kinds = [], Counter()
    for item in rows:
        gold = (item.get("label") or {}).get("tool_calls") or []
        rec = pred.get(item["id"])
        sub = rec["tool_calls"] if rec else []
        hash_ok = None
        if gold_hash is not None and item["id"] in gold_hash:
            hash_ok = bool(rec) and rec.get("state_hash") == gold_hash[item["id"]]
        r = score_sample(sub, gold, fuzzy, hash_ok)
        r["id"] = item["id"]
        if not gold and sub:
            kinds["out_of_scope_sai"] += 1
        if gold and not sub:
            kinds["bo_sot_toan_bo"] += 1
        if any(c.get("name") == "sys.ask_missing_info" for c in gold) and r["score"] == 0:
            kinds["ask_missing_info_sai"] += 1
        if hash_ok is False:
            kinds["hash_sai"] += 1
        per.append(r)
    n = max(len(per), 1)
    return {"n": len(per), "final_score": 100 * sum(x["score"] for x in per) / n,
            "avg_P": sum(x["P"] for x in per) / n, "avg_R": sum(x["R"] for x in per) / n,
            "hash_checked": gold_hash is not None, "errors": dict(kinds), "per_sample": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--gold-hash")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", help="ghi JSON chi tiết ra file")
    a = ap.parse_args()

    root = resolve_layout(a.data_dir, a.split)
    sd = root / a.split
    fuzzy = fuzzy_params(load_catalog(sd / f"{a.split}_tools_catalog.json"))
    rows = [json.loads(l) for l in open(sd / f"{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.self_test:
        pred = {r["id"]: {"tool_calls": (r.get("label") or {}).get("tool_calls") or []}
                for r in rows}
    else:
        pred = {}
        for l in open(a.pred, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                pred[d["id"]] = d
    gh = None
    if a.gold_hash:
        gh = {}
        for l in open(a.gold_hash, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                gh[d["id"]] = d["state_hash"]
    res = evaluate(pred, rows, fuzzy, gh)
    if a.report:
        Path(a.report).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_sample"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

