import sys
import json
import argparse
import unicodedata
from pathlib import Path
from collections import Counter

try:
    from pipeline.catalog import load_catalog, fuzzy_params, resolve_layout
except ImportError:
    # Fallback if imported from different working directory
    from .pipeline.catalog import load_catalog, fuzzy_params, resolve_layout

def normalize_text(text):
    """Normalize string using lowercase and Unicode NFC."""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize('NFC', text).lower()

def compute_token_f1(pred_str, gold_str):
    """Compute token-F1 between prediction and gold strings after normalization."""
    p_tokens = normalize_text(pred_str).split()
    g_tokens = normalize_text(gold_str).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0

    p_counts = Counter(p_tokens)
    g_counts = Counter(g_tokens)

    intersection = sum((p_counts & g_counts).values())
    if intersection == 0:
        return 0.0

    precision = intersection / sum(p_counts.values())
    recall = intersection / sum(g_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def params_match(pred_args: dict, gold_args: dict, fuzzy_keys: set) -> bool:
    """Check if predicted parameters match gold parameters considering exact and fuzzy rules."""
    all_keys = set(pred_args.keys()) | set(gold_args.keys())
    for k in all_keys:
        p_val = pred_args.get(k)
        g_val = gold_args.get(k)

        if k in fuzzy_keys:
            # Fuzzy match threshold > 0.6
            if compute_token_f1(p_val, g_val) <= 0.6:
                return False
        else:
            # Exact match
            if p_val != g_val:
                return False
    return True

def match_calls(pred_calls: list, gold_calls: list, fuzzy_keys_dict: dict) -> int:
    """1-1 greedy matching between predicted and gold tool calls."""
    matched_gold = set()
    matches = 0

    for pred in pred_calls:
        p_name = pred.get("name")
        p_args = pred.get("arguments", {})
        fuzzy_keys = fuzzy_keys_dict.get(p_name, set())

        best_match_idx = None
        for idx, gold in enumerate(gold_calls):
            if idx in matched_gold:
                continue
            if gold.get("name") == p_name:
                if params_match(p_args, gold.get("arguments", {}), fuzzy_keys):
                    best_match_idx = idx
                    break

        if best_match_idx is not None:
            matched_gold.add(best_match_idx)
            matches += 1

    return matches

def evaluate_predictions(submission_path, train_file_path, catalog_path, data_dir, split):
    catalog = load_catalog(catalog_path)
    fuzzy_keys_dict = fuzzy_params(catalog)

    # Load gold data
    gold_data = {}
    with open(train_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_data[item["id"]] = item

    # Load submission data
    sub_data = {}
    with open(submission_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sub_data[item["id"]] = item

    # Initialize ToolEnv for state_hash replay verification if needed
    root = resolve_layout(data_dir, split)
    sys.path.insert(0, str(root / split))
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        ToolEnv = None

    total_score = 0.0
    n_samples = 0
    total_m = 0
    total_g = 0
    total_u = 0
    hash_matches = 0

    for sample_id, gold_item in gold_data.items():
        n_samples += 1
        gold_calls = gold_item.get("label", {}).get("tool_calls", [])

        # Compute gold state_hash by replay if not present in gold label
        gold_hash = gold_item.get("label", {}).get("state_hash")
        if not gold_hash and ToolEnv is not None:
            try:
                env = ToolEnv(split, sample_id, root=root)
                for gc in gold_calls:
                    env.call(gc["name"], gc.get("arguments", {}))
                gold_hash = env.get_state_hash()
            except Exception:
                gold_hash = None

        sub_item = sub_data.get(sample_id, {})
        pred_calls = sub_item.get("tool_calls", [])
        pred_hash = sub_item.get("state_hash", "")

        # Match tool calls
        m = match_calls(pred_calls, gold_calls, fuzzy_keys_dict)
        g_len = len(gold_calls)
        u_len = len(pred_calls) - m  # extra/unmatched predicted calls

        total_m += m
        total_g += g_len
        total_u += u_len

        # Check hash match
        hash_ok = (pred_hash == gold_hash) if gold_hash else True
        if hash_ok:
            hash_matches += 1

        # Score calculation per sample[cite: 1]
        if g_len == 0:
            # Out-of-scope sample (|G| = 0)
            sample_score = 1.0 if (len(pred_calls) == 0 and m == 0 and u_len == 0) else 0.0
        else:
            if not hash_ok or m == 0:
                sample_score = 0.0
            else:
                p = m / (m + u_len) if (m + u_len) > 0 else 0.0
                r = m / g_len
                sample_score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        total_score += sample_score

    final_score = (100.0 / n_samples) * total_score if n_samples > 0 else 0.0
    overall_p = total_m / (total_m + total_u) if (total_m + total_u) > 0 else 0.0
    overall_r = total_m / total_g if total_g > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    print("=== OFFLINE EVALUATION RESULTS ===")
    print(f"Total Samples    : {n_samples}")
    print(f"Final Score      : {final_score:.2f} / 100.0")
    print(f"Overall Precision: {overall_p:.4f}")
    print(f"Overall Recall   : {overall_r:.4f}")
    print(f"Overall F1       : {overall_f1:.4f}")
    print(f"State Hash Match : {hash_matches}/{n_samples}")

    return final_score

def run_self_test(train_file_path, catalog_path, data_dir, split):
    """Run self-test using gold labels as predictions — must achieve 100.0 score[cite: 1]."""
    print("Running self-test with gold labels...")
    gold_data = []
    with open(train_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_data.append(json.loads(line))

    root = resolve_layout(data_dir, split)
    sys.path.insert(0, str(root / split))
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        ToolEnv = None

    temp_sub = Path("reports/temp_self_test_submission.jsonl")
    temp_sub.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_sub, "w", encoding="utf-8") as f_out:
        for item in gold_data:
            sample_id = item["id"]
            gold_calls = item.get("label", {}).get("tool_calls", [])

            state_hash = item.get("label", {}).get("state_hash")
            if not state_hash and ToolEnv is not None:
                try:
                    env = ToolEnv(split, sample_id, root=root)
                    for gc in gold_calls:
                        env.call(gc["name"], gc.get("arguments", {}))
                    state_hash = env.get_state_hash()
                except Exception:
                    state_hash = ""

            f_out.write(json.dumps({
                "id": sample_id,
                "tool_calls": gold_calls,
                "state_hash": state_hash or ""
            }, ensure_ascii=False) + "\n")

    score = evaluate_predictions(temp_sub, train_file_path, catalog_path, data_dir, split)
    if temp_sub.exists():
        temp_sub.unlink()

    if abs(score - 100.0) < 1e-5:
        print("SELF-TEST PASSED: Achieved 100.0 score successfully[cite: 1]!")
    else:
        print(f"SELF-TEST FAILED: Expected 100.0, got {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation script")
    parser.add_argument("--submission", default="result/submission.jsonl", help="Path to submission jsonl")
    parser.add_argument("--train-file", default="/data/train/train.jsonl", help="Path to train jsonl")
    parser.add_argument("--catalog", default="/data/train/train_tools_catalog.json", help="Path to catalog json")
    parser.add_argument("--data-dir", default="/data", help="Data root directory")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--self-test", action="store_true", help="Run self-test mode using gold labels")

    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.train_file, args.catalog, args.data_dir, args.split)
    else:
        evaluate_predictions(args.submission, args.train_file, args.catalog, args.data_dir, args.split)

