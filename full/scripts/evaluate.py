"""Chấm submission trên train: python scripts/evaluate.py --data <train+public_test> --split train --submission out.jsonl

Hash vàng = hash sau khi replay gold tool_calls trên ToolEnv của sample (qua API hợp lệ, không đọc state.json).
In điểm theo công thức đề, tỷ lệ khớp hash, điểm theo nhóm (thực thi / hỏi thêm / ngoài phạm vi).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.agent.prompt import Catalog  # noqa: E402
from lifemate.data import import_toolenv, load_catalog_raw, locate_split, read_jsonl  # noqa: E402
from lifemate.scoring import score_sample  # noqa: E402


def gold_hash(ToolEnv, split, root, item):
    env = ToolEnv(split, item["id"], root=str(root))
    for c in item["label"]["tool_calls"]:
        env.call(c["name"], c.get("arguments", {}))
    return env.get_state_hash()


def category(gold):
    if not gold:
        return "out_of_scope"
    if any(c["name"] == "sys.ask_missing_info" for c in gold):
        return "ask_missing"
    return "execute"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--show-fails", type=int, default=15)
    a = ap.parse_args()

    split, sdir, root = locate_split(a.data, a.split)
    ToolEnv = import_toolenv(sdir)
    fuzzy = Catalog(load_catalog_raw(sdir, split)).fuzzy_params()
    items = {r["id"]: r for r in read_jsonl(sdir / f"{split}.jsonl")}
    subs = {r["id"]: r for r in read_jsonl(__import__("pathlib").Path(a.submission))}

    by_cat = defaultdict(list)
    all_scores, hash_ok_n, fails = [], 0, []
    for sid, sub in subs.items():
        item = items[sid]
        gold = item["label"]["tool_calls"]
        ok = sub["state_hash"] == gold_hash(ToolEnv, split, root, item)
        hash_ok_n += ok
        r = score_sample(sub["tool_calls"], gold, ok, fuzzy)
        all_scores.append(r["score"])
        by_cat[category(gold)].append(r["score"])
        if r["score"] < 1.0:
            fails.append((sid, r, item.get("audio_text", ""), sub["tool_calls"], gold))
    n = len(all_scores)
    print(f"mẫu: {n} | điểm cuối = {100 * sum(all_scores) / n:.2f} | khớp hash: {hash_ok_n / n:.3f}")
    for c, v in sorted(by_cat.items()):
        print(f"  {c:13s} n={len(v):4d}  điểm TB={100 * sum(v) / len(v):.2f}")
    for sid, r, text, sub, gold in fails[: a.show_fails]:
        print(f"\n[{sid}] score={r['score']:.2f} M={r['M']} U={r['U']} G={r['G']} hash_ok={r['hash_ok']}")
        print("  text:", text)
        print("  sub :", json.dumps(sub, ensure_ascii=False)[:400])
        print("  gold:", json.dumps(gold, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()

