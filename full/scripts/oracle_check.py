"""Kiểm thử harness KHÔNG cần GPU/model: phát lại gold tool_calls bằng ScriptedLLM qua đúng vòng lặp agent.

Kỳ vọng: điểm ~100 và hash khớp 100%. Nếu thấp hơn nghĩa là harness/scoring/prompt có lỗi (không phải lỗi model).
  python scripts/oracle_check.py --data <train+public_test> --limit 200
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.agent.agent import ScriptedLLM, run_agent  # noqa: E402
from lifemate.agent.prompt import Catalog  # noqa: E402
from lifemate.data import import_toolenv, load_catalog_raw, locate_split, read_jsonl  # noqa: E402
from lifemate.runner import gold_transcripts  # noqa: E402
from lifemate.scoring import score_sample  # noqa: E402
from scripts.evaluate import category, gold_hash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    split, sdir, root = locate_split(a.data, "train")
    ToolEnv = import_toolenv(sdir)
    catalog = Catalog(load_catalog_raw(sdir, split))
    fuzzy = catalog.fuzzy_params()
    items = list(read_jsonl(sdir / "train.jsonl"))[: a.limit]
    scores, bad = [], []
    for it in items:
        gold = it["label"]["tool_calls"]
        env = ToolEnv(split, it["id"], root=str(root))
        tr = gold_transcripts(it)
        info = run_agent(ScriptedLLM(gold), env, catalog, it["context"], tr["text"], tr["history"])
        ok = env.get_state_hash() == gold_hash(ToolEnv, split, root, it)
        r = score_sample(env.get_tool_calls(), gold, ok, fuzzy)
        scores.append(r["score"])
        if r["score"] < 1:
            bad.append((it["id"], category(gold), info["stop"], r))
    print(f"oracle: {len(scores)} mẫu, điểm TB = {100 * sum(scores) / len(scores):.2f}")
    for b in bad[:10]:
        print("  lệch:", b)


if __name__ == "__main__":
    main()

