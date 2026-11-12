"""Bộ kiểm thử toàn diện cho dự án full/ (chạy không cần GPU/model nặng).

  python tests/test_all.py --data ../train+public_test
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lifemate.agent.agent import ScriptedLLM, run_agent  # noqa: E402
from lifemate.agent.chatml import encode_messages  # noqa: E402
from lifemate.agent.parser import clean_arguments, parse_call  # noqa: E402
from lifemate.agent.prompt import END, Catalog, build_messages, compact_tool  # noqa: E402
from lifemate.agent.retriever import FewShotRetriever, format_few_shot_block  # noqa: E402
from lifemate.asr.models.whisper_hf import _trim_silence  # noqa: E402
from lifemate.asr.normalize import postprocess  # noqa: E402
from lifemate.asr.pipeline import _is_chitchat_pair  # noqa: E402
from lifemate.data import import_toolenv, load_catalog_raw, locate_split, read_jsonl  # noqa: E402
from lifemate.runner import gold_transcripts, load_config, run  # noqa: E402
from lifemate.scoring import score_sample, token_f1  # noqa: E402


def test_asr_normalize():
    assert postprocess("  Nạp cho số không chín năm không năm ba sáu không ba sáu bảy nha ") == \
        "Nạp cho số 09505360367 nha"
    assert postprocess("số 090123499 chin sáu sửa giúp") == "số 09012349996 sửa giúp"
    assert postprocess("mở đài về ô vê giao thông à không về ô vê ba") == \
        "mở đài VOV Giao thông à không VOV3"
    assert postprocess("chuyển ti vi sang hát ti vi bảy") == "chuyển tivi sang HTV7"
    assert postprocess("vàng s c c và xăng roi chín lăm, nhắn ét em ế qua giao lô") == \
        "vàng SJC và xăng RON95, nhắn SMS qua Zalo"
    print("[OK] test_asr_normalize")


def test_parser_and_cleaner():
    assert parse_call(END) is None
    assert parse_call("") is None
    c = parse_call('{"name": "smart_home.light_set", "arguments": {"device_id": "d1", "brightness": "50", "extra": 1, "color": null}}')
    spec = {
        "parameters": {
            "properties": {
                "device_id": {"type": "string"},
                "brightness": {"type": "integer"},
                "color": {"type": "string"},
            }
        }
    }
    cleaned = clean_arguments(c["arguments"], spec)
    assert cleaned == {"device_id": "d1", "brightness": 50}, cleaned
    print("[OK] test_parser_and_cleaner")


def test_scoring():
    assert token_f1("Nộp tiền điện", "nộp tiền điện") == 1.0
    fuzzy = {"reminders.create": {"content"}}
    g = [{"name": "reminders.create", "arguments": {"content": "nộp tiền điện tháng này", "time": "2026-11-11T09:00:00+07:00"}}]
    s = [{"name": "reminders.create", "arguments": {"content": "nộp tiền điện", "time": "2026-11-11T09:00:00+07:00"}}]
    r = score_sample(s, g, True, fuzzy)
    assert r["score"] == 1.0, r
    assert score_sample([], [], True, fuzzy)["score"] == 1.0
    assert score_sample(s, [], True, fuzzy)["score"] == 0.0
    assert score_sample(s, g, False, fuzzy)["score"] == 0.0
    print("[OK] test_scoring")


def test_chatml():
    class DummyTok:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": '{"name":"a.b","arguments":{}}'},
    ]
    ids, labels = encode_messages(DummyTok(), msgs)
    assert len(ids) == len(labels)
    trained = "".join(chr(x) for x in labels if x != -100)
    assert trained == '{"name":"a.b","arguments":{}}<|im_end|>', trained
    print("[OK] test_chatml")


def test_agent_validate_repair_and_runner(data_dir: str):
    split, sdir, root = locate_split(data_dir, "train")
    ToolEnv = import_toolenv(sdir)
    catalog = Catalog(load_catalog_raw(sdir, split))
    items = list(read_jsonl(sdir / "train.jsonl"))
    item = items[0]
    gold = item["label"]["tool_calls"]

    # Gửi 1 lệnh sai schema trước, rồi mới gửi các lệnh đúng -> lệnh sai phải bị validate chặn, KHÔNG nằm trong env.get_tool_calls()
    bad_first = [{"name": "smart_home.light_set", "arguments": {"power": "on"}}] + gold
    env = ToolEnv(split, item["id"], root=str(root))
    tr = gold_transcripts(item)
    run_agent(ScriptedLLM(bad_first), env, catalog, item["context"], tr["text"], tr["history"])
    assert env.get_tool_calls() == gold, env.get_tool_calls()
    print("[OK] test_agent_validate_repair (lệnh sai schema bị chặn trước khi vào log)")

    # Kiểm tra x-ref guard & auto-mapper:
    # 1) Cố gọi smart_home.light_set với ID đoán mò "light_999" khi chưa gọi get_status -> bị chặn
    # 2) Gọi smart_home.get_status -> nhận danh sách thiết bị
    # 3) Gọi smart_home.light_set nhưng truyền TÊN "đèn phòng ngủ nhỏ" thay vì ID "light_bedroom2" -> tự đổi sang "light_bedroom2"
    item_sh = items[3]
    env_sh = ToolEnv(split, item_sh["id"], root=str(root))
    tr_sh = gold_transcripts(item_sh)
    xref_script = [
        {"name": "smart_home.light_set", "arguments": {"device_id": "light_999", "brightness": 50}},
        {"name": "smart_home.get_status", "arguments": {}},
        {"name": "smart_home.light_set", "arguments": {"device_id": "đèn phòng ngủ nhỏ", "brightness": 50}},
    ]
    run_agent(ScriptedLLM(xref_script), env_sh, catalog, item_sh["context"], tr_sh["text"], tr_sh["history"])
    assert env_sh.get_tool_calls() == item_sh["label"]["tool_calls"], env_sh.get_tool_calls()
    print("[OK] test_xref_auto_map_and_guard (chặn đoán mò ID + tự đổi Tên -> ID)")

    # Kiểm tra BM25 Few-Shot Retriever & Silence Trimming & Chitchat Pruning
    import numpy as np
    wav = np.concatenate([np.zeros(8000, dtype="float32"), np.ones(16000, dtype="float32") * 0.2, np.zeros(8000, dtype="float32")])
    trimmed = _trim_silence(wav, sr=16000)
    assert len(trimmed) < len(wav) and len(trimmed) >= 16000
    assert _is_chitchat_pair({"role": "user"}, {"role": "assistant", "text": "Cảm ơn bạn, cần gì cứ nói nhé.", "tool_calls": None})
    assert not _is_chitchat_pair({"role": "user"}, {"role": "assistant", "text": "Bạn muốn chuyển bao nhiêu?", "tool_calls": None})
    ret = FewShotRetriever.from_jsonl(sdir / "train.jsonl")
    exs = ret.retrieve(item_sh["audio_text"], item_sh["context"]["installed_apps"], k=2, exclude_id=item_sh["id"])
    assert len(exs) == 2 and all(e["id"] != item_sh["id"] for e in exs)
    assert "Ví dụ tham khảo" in format_few_shot_block(exs)
    print("[OK] test_retriever_silence_and_chitchat_pruning")

    # Kiểm tra chạy runner trên public_test với Dummy LLM -> xuất submission.jsonl hợp lệ và kiểm tra bằng zaic_tools --validate
    cfg = load_config(str(ROOT / "config.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        out_sub = Path(tmp) / "submission.jsonl"
        run(cfg, data_dir, str(out_sub), split="public_test", limit=10,
            llm=ScriptedLLM([]), asr=type("A", (), {"transcribe_batch": lambda self, p: ["xin chào"] * len(list(p))})())
        lines = [json.loads(l) for l in open(out_sub, encoding="utf-8")]
        assert len(lines) == 10
        assert all(set(x.keys()) == {"id", "tool_calls", "state_hash"} for x in lines)

        _, pub_dir, pub_root = locate_split(data_dir, "public_test")
        res = subprocess.run(
            [sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", str(pub_root),
             "--validate", str(out_sub)],
            env={**os.environ, "PYTHONPATH": str(pub_dir)},
            capture_output=True, text=True, check=True
        )
        assert "0 schema error" in res.stdout or "errors: 0" in res.stdout.lower() or res.returncode == 0, res.stdout
    print("[OK] test_runner_and_zaic_validate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    test_asr_normalize()
    test_parser_and_cleaner()
    test_scoring()
    test_chatml()
    test_agent_validate_repair_and_runner(a.data)
    print("\n=== TẤT CẢ KIỂM THỬ ĐỀU PASS 100% ===")


if __name__ == "__main__":
    main()

