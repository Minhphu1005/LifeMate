"""Truy xuất ví dụ mẫu động (Dynamic Few-Shot Retrieval) bằng BM25 + lọc theo installed_apps + tự học từ history của split.

Giúp LLM (kể cả khi chạy Zero-shot chưa fine-tune lẫn khi đã fine-tune LoRA):
1. Nhìn thấy 2-3 bài giải mẫu có cấu trúc câu và tổ hợp ứng dụng giống nhất với câu yêu cầu hiện tại.
2. Tự động khai thác các cặp (user/assistant, tool_calls) có sẵn trong trường `history` của tập dữ liệu đang chạy
   (public_test / private_test) để có ngay ví dụ mẫu thực tế cho các ứng dụng mới chưa từng xuất hiện ở tập train.
3. Với ứng dụng hoàn toàn mới chưa có mẫu nào trong kho, tự động sinh gợi ý luồng gọi (Read -> Write theo x-ref)
   từ Catalog và tránh đưa ví dụ của ứng dụng khác dễ gây nhiễu.
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
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


class FewShotRetriever:
    def __init__(self, records: List[dict]):
        """records: [{"id", "text", "apps": list[str], "calls": list[dict], "source_id": optional str}]"""
        self.records: List[dict] = []
        self.docs: List[List[str]] = []
        self.tf: List[Counter] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.N = 1
        self.avgdl = 1.0
        self._seen_keys: Set[str] = set()
        self.add_records(records)

    def _rebuild_stats(self):
        self.N = max(len(self.records), 1)
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.docs else 1.0

    def add_records(self, new_records: List[dict]):
        added = 0
        for r in new_records:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            calls = r.get("calls") or []
            key = f"{r.get('id', '')}|{txt[:80]}|{json.dumps(calls, ensure_ascii=False, sort_keys=True)}"
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            toks = _tokenize(txt)
            if not toks:
                continue
            self.records.append(r)
            self.docs.append(toks)
            self.tf.append(Counter(toks))
            for w in set(toks):
                self.doc_freqs[w] += 1
            added += 1
        if added:
            self._rebuild_stats()

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
                        "source_id": r["id"],
                        "text": r.get("audio_text", ""),
                        "apps": apps,
                        "installed": sorted((r.get("context") or {}).get("installed_apps", {}).keys()),
                        "calls": calls,
                    })
                elif "calls" in r:
                    if "source_id" not in r:
                        r["source_id"] = str(r.get("id", "")).split("#")[0]
                    records.append(r)
        return cls(records)

    def ingest_split_histories(self, items: List[dict], asr_cache: Optional[Dict[str, dict]] = None):
        """Khai thác các lượt history có tool_calls hợp lệ từ danh sách mẫu của split hiện tại (public_test/private_test)."""
        new_recs = []
        for it in items:
            sid = it.get("id", "")
            hist = it.get("history") or []
            cached_hist = (asr_cache or {}).get(sid, {}).get("history") if asr_cache else None
            for i, t in enumerate(hist):
                if t.get("role") != "assistant":
                    continue
                calls = t.get("tool_calls") or []
                if not calls:
                    continue
                # Bỏ qua các bước gọi lỗi (done=False) trong history nếu có
                results = t.get("tool_results") or []
                if any(isinstance(res, dict) and res.get("done") is False for res in results):
                    continue
                user_txt = ""
                if i > 0 and hist[i - 1].get("role") == "user":
                    user_txt = hist[i - 1].get("audio_text") or hist[i - 1].get("text") or ""
                    if not user_txt and cached_hist and i - 1 < len(cached_hist):
                        user_txt = cached_hist[i - 1].get("text") or ""
                asst_txt = t.get("text") or ""
                # Nếu chưa có transcript của lượt user trong history, dùng lời phản hồi của trợ lý + tham số làm văn bản truy xuất
                arg_words = " ".join(
                    str(v) for c in calls for v in (c.get("arguments") or {}).values() if isinstance(v, (str, int))
                )
                combined_text = user_txt.strip() if user_txt.strip() else f"{asst_txt} {arg_words}".strip()
                if not combined_text:
                    continue
                apps = sorted({c["name"].split(".")[0] for c in calls if "." in c.get("name", "")})
                new_recs.append({
                    "id": f"{sid}#h{i}",
                    "source_id": sid,
                    "text": combined_text,
                    "apps": apps,
                    "calls": calls,
                    "from_history": not bool(user_txt.strip()),
                })
        self.add_records(new_recs)

    def save_compact(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def retrieve(self, query_text: str, installed_apps: dict, k: int = 2,
                 exclude_id: Optional[str] = None, catalog=None) -> List[dict]:
        if k <= 0 or not self.records or not query_text.strip():
            return []
        inst: Set[str] = set(installed_apps.keys())
        q_toks = _tokenize(query_text)
        if not q_toks:
            return []
        q_set = set(q_toks)

        # Nhận diện các app trong installed_apps có từ khóa mô tả khớp trực tiếp với câu hỏi
        candidate_apps: Set[str] = set()
        if catalog is not None:
            for app_id in inst:
                meta = catalog.apps.get(app_id, {})
                app_desc = f"{app_id} {meta.get('name_vi', '')} " + " ".join(
                    (catalog.tools[tn].get("description") or "")
                    for tn in installed_apps.get(app_id, {}).get("tools", [])
                    if tn in catalog.tools
                )
                app_toks = set(_tokenize(app_desc))
                # Đếm số bigram hoặc từ khóa đặc trưng trùng nhau
                overlap = q_set & app_toks
                if any("_" in w for w in overlap) or len(overlap) >= 2:
                    candidate_apps.add(app_id)

        k1, b = 1.5, 0.75
        scored = []
        for idx, rec in enumerate(self.records):
            src_id = rec.get("source_id") or str(rec.get("id", "")).split("#")[0]
            if exclude_id and (rec["id"] == exclude_id or src_id == exclude_id):
                continue
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
                # Thưởng điểm nếu ví dụ gọi đúng ứng dụng ứng viên trong catalog
                if candidate_apps and (rec_apps & candidate_apps):
                    score *= 1.35
                # Ưu tiên câu yêu cầu trực tiếp của người dùng hơn câu tóm tắt của trợ lý
                if rec.get("from_history"):
                    score *= 0.85
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
        label_prefix = "Ngữ cảnh tương tự" if ex.get("from_history") else "Yêu cầu"
        lines.append(f"- {label_prefix}: \"{ex['text']}\"\n  Luồng lệnh mẫu: {chain}")
    return "\n".join(lines)
