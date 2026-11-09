"""Test end-to-end offline bằng dataset + zaic_tools GIẢ (không cần dữ liệu thật/model).

  python -m unittest tests.test_pipeline_mock -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAKE_ZAIC = '''
import hashlib, json
class ToolEnv:
    def __init__(self, split, sample_id, root="."):
        self.log, self.writes = [], []
    def validate(self, name, args):
        return {"ok": True}
    def call(self, name, args):
        self.log.append({"name": name, "arguments": args})
        if name == "boom.fail":
            return {"ok": False, "error": "x"}
        if not name.endswith("get_status"):
            self.writes.append(name + json.dumps(args, sort_keys=True))
        return {"ok": True, "devices": [{"id": "d1"}]}
    def get_tool_calls(self):
        return list(self.log)
    def get_state_hash(self):
        return hashlib.sha1("".join(self.writes).encode()).hexdigest()[:16]
'''

CATALOG = {"apps": [{"id": "smart_home", "tools": [
    {"name": "smart_home.get_status", "kind": "read", "parameters": {"type": "object", "properties": {}}},
    {"name": "smart_home.light_set", "kind": "write", "parameters": {"type": "object", "properties": {
        "device_id": {"type": "string", "x-ref": "smart_home.devices"},
        "power": {"type": "string", "enum": ["on", "off"]},
        "note": {"type": "string", "x-match": "fuzzy"}}, "required": ["device_id"]}},
    {"name": "smart_home.unused", "kind": "write", "parameters": {"type": "object", "properties": {}}},
]}, {"id": "sys", "tools": [
    {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object", "properties": {
        "tool_name": {"type": "string"}, "params": {"type": "array"}}}}]}]}

GOLD = [{"name": "smart_home.get_status", "arguments": {}},
        {"name": "smart_home.light_set", "arguments": {"device_id": "d1", "power": "off"}}]


def make_root(tmp: Path, split="train"):
    sd = tmp / split
    (sd / "zaic_tools").mkdir(parents=True)
    (sd / "zaic_tools" / "__init__.py").write_text(FAKE_ZAIC, encoding="utf-8")
    (sd / f"{split}_tools_catalog.json").write_text(json.dumps(CATALOG), encoding="utf-8")
    ctx = {"current_time": "2026-11-16T17:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
           "user_profile": {}, "installed_apps": {"smart_home": {"tools": [
               "smart_home.get_status", "smart_home.light_set", "smart_home.nope"]}}}
    rows = [{"id": f"{split}_{i}", "audio": f"audio/{split}_{i}/{split}_{i}.wav", "context": ctx,
             "history": [], "label": {"tool_calls": GOLD if i == 0 else []}} for i in range(2)]
    with open(sd / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp


class WriterLLM:
    """Đọc 1 read qua env_caller, rồi trả cả read lẫn write (read KHÔNG được chạy lại)."""
    def generate_calls(self, user_text, history, user_context, tools, env_caller=None, validator=None):
        env_caller("smart_home.get_status", {})
        try:
            env_caller("smart_home.light_set", {"device_id": "d1"})   # phải bị chặn
        except PermissionError:
            pass
        return [GOLD[0], GOLD[1], {"name": "not.in.list", "arguments": {}}]


class CrashLLM:
    def generate_calls(self, *a, **k):
        raise RuntimeError("hỏng")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(Path(self.tmp.name))
        for m in [m for m in sys.modules if m.startswith("zaic_tools")]:
            del sys.modules[m]

    def tearDown(self):
        self.tmp.cleanup()
        sys.path = [p for p in sys.path if not p.endswith(str(Path("train")))]

    def _run(self, llm_obj):
        from pipeline import runner
        orig = runner.build_components
        runner.build_components = lambda cfg: (__import__("asr.models.asr_factory", fromlist=["x"])
                                               .build_asr({"engine": "dummy"}), llm_obj)
        try:
            out = self.root / "out" / "submission.jsonl"
            cfg = {"system": {"seed": 1, "timeout_per_sample": 5}, "paths": {"reports": str(self.root / "rep")}}
            runner.run(cfg, "train", self.root, out)
        finally:
            runner.build_components = orig
        with open(out, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        return recs, self.root / "rep"

    def test_dummy_llm_gives_empty_calls(self):
        from pipeline.stubs import DummyLLM
        recs, rep = self._run(DummyLLM())
        self.assertEqual([r["id"] for r in recs], ["train_0", "train_1"])
        self.assertTrue(all(r["tool_calls"] == [] and r["state_hash"] is not None for r in recs))
        self.assertTrue((rep / "latency.csv").exists() and (rep / "env_info.txt").exists())

    def test_no_duplicate_read_and_write_executed(self):
        recs, rep = self._run(WriterLLM())
        calls = [c["name"] for c in recs[0]["tool_calls"]]
        self.assertEqual(calls, ["smart_home.get_status", "smart_home.light_set"])
        with open(rep / "failures.jsonl", encoding="utf-8") as f:
            fails = f.read()
        self.assertIn("nope", fails)            # tool ngoài catalog được cảnh báo
        self.assertIn("không hợp lệ", fails)    # call not.in.list bị bỏ

    def test_llm_crash_still_one_line_per_sample(self):
        recs, rep = self._run(CrashLLM())
        self.assertEqual(len(recs), 2)
        with open(rep / "failures.jsonl", encoding="utf-8") as f:
            self.assertIn("hỏng", f.read())

    def test_evaluate_gold_is_100_and_partial_less(self):
        from evaluate import evaluate
        from pipeline.catalog import fuzzy_params, load_catalog
        from pipeline.verify_data import verify
        sd = self.root / "train"
        fz = fuzzy_params(load_catalog(sd / "train_tools_catalog.json"))
        with open(sd / "train.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        v = verify(str(self.root), "train")
        self.assertIn("point_1_write_to_write", v)
        self.assertIn("point_2_ask_missing_info", v)
        gold = {r["id"]: {"tool_calls": r["label"]["tool_calls"]} for r in rows}
        self.assertAlmostEqual(evaluate(gold, rows, fz)["final_score"], 100.0)
        part = {"train_0": {"tool_calls": [GOLD[1]]}, "train_1": {"tool_calls": []}}
        self.assertLess(evaluate(part, rows, fz)["final_score"], 100.0)
        wrong_oos = {"train_0": {"tool_calls": GOLD}, "train_1": {"tool_calls": [GOLD[0]]}}
        self.assertEqual(evaluate(wrong_oos, rows, fz)["final_score"], 50.0)
        bad_hash = evaluate(gold, rows, fz, {"train_0": "x", "train_1": "y"})
        self.assertEqual(bad_hash["final_score"], 50.0)   # train_1 OOS vẫn 1 nếu rỗng, train_0 hash sai -> 0

    def test_allow_write_in_env_caller_and_no_retry_on_fail(self):
        class WriteChainLLM:
            def generate_calls(self, user_text, history, user_context, tools, env_caller=None, validator=None):
                env_caller("smart_home.light_set", {"device_id": "d1", "power": "off"})
                return [GOLD[1]]

        from pipeline import runner
        orig = runner.build_components
        runner.build_components = lambda cfg: (__import__("asr.models.asr_factory", fromlist=["x"])
                                               .build_asr({"engine": "dummy"}), WriteChainLLM())
        try:
            out = self.root / "out" / "sub_w.jsonl"
            cfg = {"system": {"seed": 1, "timeout_per_sample": 5, "allow_write_in_env_caller": True},
                   "paths": {"reports": str(self.root / "rep_w")}}
            runner.run(cfg, "train", self.root, out)
        finally:
            runner.build_components = orig
        with open(out, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        # write gọi qua env_caller không bị chạy lại lần 2 ở execute_rest
        self.assertEqual(len(recs[0]["tool_calls"]), 1)
        self.assertEqual(recs[0]["tool_calls"][0]["name"], "smart_home.light_set")


if __name__ == "__main__":
    unittest.main()

