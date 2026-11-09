"""Script kiểm tra nhanh trên train.jsonl cho 2 rủi ro logic:
  1) Có mẫu nào chứa chuỗi write -> write phụ thuộc ID của nhau không?
     (để quyết định bật/tắt system.allow_write_in_env_caller trong config.yaml)
  2) Các mẫu có sys.ask_missing_info trong Gold có kèm read call trước đó không?

Chạy:
  python -m pipeline.verify_data --data-dir <ROOT> --split train
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Set

from .catalog import ASK, load_catalog, resolve_layout


def _collect_strings(obj: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(obj, str) and obj:
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= _collect_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _collect_strings(v)
    return out


def verify(data_dir: str, split: str = "train") -> dict:
    root = resolve_layout(data_dir, split)
    sd = root / split
    catalog = load_catalog(sd / f"{split}_tools_catalog.json")
    kind = {name: t.get("kind") for name, t in catalog.items()}

    with open(sd / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    write_chains = []
    write_id_deps = []
    ask_total = 0
    ask_only = 0
    ask_with_read = []
    ask_patterns = Counter()

    for r in rows:
        label = r.get("label") or {}
        calls = label.get("tool_calls") or []
        results = label.get("tool_results") or []
        kinds = [kind.get(c.get("name"), "?") for c in calls]

        # 1) Kiểm tra nhiều write trong cùng 1 mẫu và phụ thuộc kết quả write trước
        write_indices = [i for i, k in enumerate(kinds) if k == "write"]
        if len(write_indices) >= 2:
            write_chains.append((r["id"], [c["name"] for c in calls]))
            for idx_pos, wi in enumerate(write_indices[:-1]):
                res_strs = _collect_strings(results[wi]) if wi < len(results) else set()
                for wj in write_indices[idx_pos + 1:]:
                    arg_strs = _collect_strings(calls[wj].get("arguments") or {})
                    shared = {s for s in (res_strs & arg_strs) if len(s) >= 2}
                    if shared:
                        write_id_deps.append({
                            "id": r["id"],
                            "from_write": calls[wi]["name"],
                            "to_write": calls[wj]["name"],
                            "shared_values": sorted(shared),
                        })

        # 2) Kiểm tra các mẫu có sys.ask_missing_info
        if any(c.get("name") == ASK for c in calls):
            ask_total += 1
            ask_patterns[tuple(kinds)] += 1
            if len(calls) == 1 and calls[0].get("name") == ASK:
                ask_only += 1
            else:
                ask_with_read.append({
                    "id": r["id"],
                    "calls": [c["name"] for c in calls],
                    "ask_args": next(c.get("arguments") for c in calls if c.get("name") == ASK),
                })

    summary = {
        "n_samples": len(rows),
        "point_1_write_to_write": {
            "samples_with_2plus_writes": len(write_chains),
            "samples_with_write_output_used_in_later_write": len(write_id_deps),
            "recommend_allow_write_in_env_caller": len(write_id_deps) > 0,
            "examples": write_id_deps[:10],
        },
        "point_2_ask_missing_info": {
            "total_ask_samples": ask_total,
            "ask_only_no_read": ask_only,
            "ask_with_preceding_calls": len(ask_with_read),
            "kind_patterns": {" -> ".join(k): v for k, v in ask_patterns.most_common()},
            "examples_with_preceding_calls": ask_with_read[:10],
        },
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", help="ghi kết quả JSON ra file")
    a = ap.parse_args()
    res = verify(a.data_dir, a.split)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

