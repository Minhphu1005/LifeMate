"""Truy xuất ví dụ mẫu động (Dynamic Few-Shot Retrieval) bằng BM25 + lọc theo installed_apps.

Giúp LLM (kể cả khi chạy Zero-shot chưa fine-tune lẫn khi đã fine-tune LoRA) nhìn thấy 2-3 bài giải mẫu
có cấu trúc câu và tổ hợp ứng dụng giống nhất với câu yêu cầu hiện tại.
"""
from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set


def _tokenize(text: str) -> List[str]:
    s = unicodedata.normalize("NFC", str(text)).lower()
    words = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s).split()
    # Kết hợp từ đơn (unigrams) và cặp từ liền kề (bigrams) để bắt cụm tiếng Việt ("điều hòa", "báo thức", "chuyển tiền")
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


class FewShotRetriever:
    def __init__(self, records: List[dict]):
        """records: [{"id", "text", "apps": list[str], "calls": list[dict]}]"""
        self.records = records
        self.docs = [_tokenize(r["text"]) for r in records]
        self.doc_freqs: Dict[str, int] = Counter()
        for d in self.docs:
            for w in set(d):
                self.doc_freqs[w] += 1
        self.N = max(len(records), 1)
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.docs else 1.0
        self.tf = [Counter(d) for d in self.docs]

    @classmethod
    def from_jsonl(cls, path: Path) -> "FewShotRetriever":
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "label" in r:
                    calls = r["label"]["tool_calls"]
                    apps = sorted({c["name"].split(".")[0] for c in calls if "." in c.get("name", "")})
                    records.append({
                        "id": r["id"],
                        "text": r.get("audio_text", ""),
                        "apps": apps,
                        "installed": sorted((r.get("context") or {}).get("installed_apps", {}).keys()),
                        "calls": calls,
                    })
                elif "calls" in r:
                    records.append(r)
        return cls(records)

    def save_compact(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def retrieve(self, query_text: str, installed_apps: dict, k: int = 2,
                 exclude_id: Optional[str] = None) -> List[dict]:
        if k <= 0 or not self.records or not query_text.strip():
            return []
        inst: Set[str] = set(installed_apps.keys())
        q_toks = _tokenize(query_text)
        if not q_toks:
            return []

        k1, b = 1.5, 0.75
        scored = []
        for idx, rec in enumerate(self.records):
            if exclude_id and rec["id"] == exclude_id:
                continue
            # Chỉ lấy ví dụ mà các app được gọi đều nằm trong installed_apps của mẫu hiện tại
            rec_apps = set(rec.get("apps") or [])
            if rec_apps and not rec_apps.issubset(inst):
                continue
            dl = len(self.docs[idx])
            if dl == 0:
                continue
            tf_map = self.tf[idx]
            score = 0.0
            for w in q_toks:
                f = tf_map.get(w, 0)
                if f == 0:
                    continue
                df = self.doc_freqs.get(w, 0)
                idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / self.avgdl))
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return [self.records[i] for _, i in scored[:k]]


def format_few_shot_block(examples: List[dict]) -> str:
    if not examples:
        return ""
    lines = [
        "Ví dụ tham khảo cách chọn luồng gọi tool (CHỈ tham khảo luồng lệnh, TUYỆT ĐỐI không chép ID từ ví dụ):"
    ]
    for ex in examples:
        calls = ex.get("calls") or []
        if not calls:
            chain = "<end> (ngoài phạm vi hoặc đã thoả điều kiện không cần làm gì)"
        else:
            steps = [json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, ensure_ascii=False)
                     for c in calls]
            if calls[-1]["name"] != "sys.ask_missing_info":
                steps.append("<end>")
            chain = " → ".join(steps)
        lines.append(f"- Yêu cầu: \"{ex['text']}\"\n  Luồng lệnh mẫu: {chain}")
    return "\n".join(lines)

