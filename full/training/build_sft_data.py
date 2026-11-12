"""Tạo dữ liệu SFT cho LLM agent từ gold trajectory của train.

  python training/build_sft_data.py --data ../train+public_test --out-dir sft_data \
      --asr-train asr_train.jsonl --holdout-apps ""

Mỗi sample thành MỘT chuỗi hội thoại nhiều lượt: system(tool+luật) / user(ngữ cảnh+yêu cầu) /
assistant(lệnh 1) / user(KẾT QUẢ 1) / ... / assistant(<end>). Loss chỉ tính trên các lượt assistant.
- Biến thể 'noisy': thay transcript chuẩn bằng transcript do ASR sinh (nếu sai ít) để LLM chịu được lỗi nhận dạng.
- --holdout-apps: đưa các mẫu dùng app đó vào tập val (mô phỏng tool chưa từng thấy, vì public_test có tool mới).
"""
import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifemate.agent.chatml import encode_messages  # noqa: E402
from lifemate.agent.prompt import END, RESULT_PREFIX, Catalog, call_json, compact_result, system_message, user_message  # noqa: E402
from lifemate.agent.retriever import FewShotRetriever, format_few_shot_block  # noqa: E402
from lifemate.data import load_catalog_raw, locate_split, read_jsonl  # noqa: E402
from lifemate.runner import gold_transcripts  # noqa: E402


def norm_words(s):
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(c if (c.isalnum() or c.isspace()) else " " for c in s).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def trajectory_messages(catalog, ctx, text, history, calls, results, few_shot_block=""):
    msgs = [{"role": "system", "content": system_message(catalog, ctx["installed_apps"])},
            {"role": "user", "content": user_message(ctx, text, history, few_shot_block=few_shot_block)}]
    for i, (c, r) in enumerate(zip(calls, results)):
        msgs.append({"role": "assistant", "content": call_json(c["name"], c.get("arguments", {}))})
        if c["name"] == "sys.ask_missing_info":
            return msgs  # inference dừng ngay sau ask_missing_info
        msgs.append({"role": "user", "content": RESULT_PREFIX + compact_result(r)})
    msgs.append({"role": "assistant", "content": END})
    return msgs


def noisy_transcripts(item, cache_row, max_wer):
    """Ghép transcript ASR (nếu sai ít) vào đúng cấu trúc history của gold."""
    gold = gold_transcripts(item)
    h_asr = [t for t in cache_row.get("history", []) if t.get("role") == "user"]
    hist, k = [], 0
    for t in gold["history"]:
        if t["role"] == "user":
            hyp = h_asr[k]["text"] if k < len(h_asr) else t["text"]
            k += 1
            hist.append({"role": "user", "text": hyp if wer(t["text"], hyp) <= max_wer else t["text"]})
        else:
            hist.append(t)
    hyp = cache_row["text"]
    if wer(gold["text"], hyp) > max_wer or not hyp.strip():
        return None
    return {"text": hyp, "history": hist}


def load_tok(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))

        class Shim:
            def encode(self, text, add_special_tokens=False):
                return tk.encode(text, add_special_tokens=False).ids
        return Shim()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="sft_data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--holdout-apps", default="", help="app cách nhau dấu phẩy; mẫu dùng app này -> val")
    ap.add_argument("--asr-train", default=None, help="jsonl transcript ASR của train (run_split)")
    ap.add_argument("--noisy-max-wer", type=float, default=0.35)
    ap.add_argument("--few-shot-k", type=int, default=2)
    ap.add_argument("--few-shot-bank", default="weights/few_shot_bank.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    split, sdir, _ = locate_split(a.data, "train")
    catalog = Catalog(load_catalog_raw(sdir, split))
    items = list(read_jsonl(sdir / "train.jsonl"))
    cache = {r["id"]: r for r in read_jsonl(Path(a.asr_train))} if a.asr_train else {}
    holdout = {x.strip() for x in a.holdout_apps.split(",") if x.strip()}

    def is_val(item):
        gold_apps = {c["name"].split(".")[0] for c in item["label"]["tool_calls"]}
        if holdout & gold_apps:
            return True
        h = int(hashlib.md5(f"{a.seed}:{item['id']}".encode()).hexdigest(), 16) % 10000
        return h < a.val_frac * 10000

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Xây dựng kho Few-Shot từ các mẫu thuộc tập train (không đưa mẫu val vào kho để tránh leak khi đo val)
    bank_records = []
    for it in items:
        if not is_val(it):
            calls = it["label"]["tool_calls"]
            apps = sorted({c["name"].split(".")[0] for c in calls if "." in c.get("name", "")})
            bank_records.append({
                "id": it["id"],
                "text": it.get("audio_text", ""),
                "apps": apps,
                "calls": calls,
            })
    retriever = FewShotRetriever(bank_records)
    if a.few_shot_bank:
        retriever.save_compact(Path(a.few_shot_bank))
        print(f"[few-shot] đã lưu {len(bank_records)} mẫu vào {a.few_shot_bank}")

    ftr, fva = open(out / "sft_train.jsonl", "w", encoding="utf-8"), open(out / "sft_val.jsonl", "w", encoding="utf-8")
    ntr = nva = nnoisy = 0
    val_ids = []
    for it in items:
        lab = it["label"]
        g = gold_transcripts(it)
        inst = it["context"].get("installed_apps", {})
        fs_gold = format_few_shot_block(retriever.retrieve(g["text"], inst, k=a.few_shot_k, exclude_id=it["id"])) if a.few_shot_k > 0 else ""
        msgs = trajectory_messages(catalog, it["context"], g["text"], g["history"], lab["tool_calls"], lab["tool_results"], few_shot_block=fs_gold)
        if is_val(it):
            fva.write(json.dumps({"id": it["id"], "variant": "gold", "messages": msgs}, ensure_ascii=False) + "\n")
            nva += 1
            val_ids.append(it["id"])
            continue
        ftr.write(json.dumps({"id": it["id"], "variant": "gold", "messages": msgs}, ensure_ascii=False) + "\n")
        ntr += 1
        if it["id"] in cache:
            nz = noisy_transcripts(it, cache[it["id"]], a.noisy_max_wer)
            if nz and nz["text"] != g["text"]:
                fs_nz = format_few_shot_block(retriever.retrieve(nz["text"], inst, k=a.few_shot_k, exclude_id=it["id"])) if a.few_shot_k > 0 else ""
                m2 = trajectory_messages(catalog, it["context"], nz["text"], nz["history"], lab["tool_calls"], lab["tool_results"], few_shot_block=fs_nz)
                ftr.write(json.dumps({"id": it["id"], "variant": "noisy", "messages": m2}, ensure_ascii=False) + "\n")
                ntr += 1; nnoisy += 1
    ftr.close(); fva.close()
    json.dump(val_ids, open(out / "val_ids.json", "w"))
    print(f"train seq={ntr} (noisy={nnoisy}) | val seq={nva}")

    tok = load_tok(a.tokenizer)
    lens = []
    with open(out / "sft_train.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 400:
                break
            ids, _ = encode_messages(tok, json.loads(line)["messages"])
            lens.append(len(ids))
    lens.sort()
    print(f"độ dài token (400 mẫu đầu): TB={sum(lens) / len(lens):.0f} p95={lens[int(.95 * len(lens))]} max={lens[-1]}")


if __name__ == "__main__":
    main()

