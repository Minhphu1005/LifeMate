import unittest
import json
import tempfile
from pathlib import Path

# Import pipeline and evaluation components
from pipeline.runner import remaining_calls, filter_tools, transcribe_sample
from pipeline.stubs import DummyLLM, ReadProbeLLM
from pipeline.catalog import fuzzy_params, compact_tool
from evaluate import compute_token_f1, params_match, match_calls

class TestPipelineMock(unittest.TestCase):

    def test_multiset_deduplication(self):
        """Test that remaining_calls properly handles multiset deduplication[cite: 1]."""
        calls = [
            {"name": "smart_home.get_status", "arguments": {"device_id": "light_1"}},
            {"name": "smart_home.get_status", "arguments": {"device_id": "light_1"}},
            {"name": "smart_home.light_set", "arguments": {"device_id": "light_1", "state": "on"}}
        ]
        executed = [
            {"name": "smart_home.get_status", "arguments": {"device_id": "light_1"}}
        ]

        remaining = remaining_calls(calls, executed)
        # Should retain one get_status and the light_set
        self.assertEqual(len(remaining), 2)
        self.assertEqual(remaining[0]["name"], "smart_home.get_status")
        self.assertEqual(remaining[1]["name"], "smart_home.light_set")

    def test_dummy_and_probe_llm(self):
        """Test stub LLMs behavior[cite: 1]."""
        dummy = DummyLLM()
        self.assertEqual(dummy.generate_calls("hi", [], {}, []), [])

        probe = ReadProbeLLM()
        tools = [
            {"name": "smart_home.get_status", "kind": "read", "parameters": {}},
            {"name": "smart_home.light_set", "kind": "write", "parameters": {"required": ["device_id"]}}
        ]
        calls = probe.generate_calls("turn on light", [], {}, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "smart_home.get_status")

    def test_token_f1_scoring(self):
        """Test fuzzy token-F1 calculation for parameters[cite: 1]."""
        # Exact/close match above threshold > 0.6
        score1 = compute_token_f1("Turn on the living room light", "turn on living room light")
        self.assertGreater(score1, 0.6)

        # Completely different text
        score2 = compute_token_f1("Play music", "turn off AC")
        self.assertEqual(score2, 0.0)

    def test_params_matching(self):
        """Test parameter matching logic with fuzzy and exact rules[cite: 1]."""
        fuzzy_keys = {"note"}

        pred = {"note": "Buy milk today", "count": 2}
        gold = {"note": "buy milk today", "count": 2}

        self.assertTrue(params_match(pred, gold, fuzzy_keys))

        # Different exact parameter 'count'
        bad_pred = {"note": "Buy milk today", "count": 3}
        self.assertFalse(params_match(bad_pred, gold, fuzzy_keys))

    def test_call_matching_algorithm(self):
        """Test greedy 1-1 tool call matching[cite: 1]."""
        fuzzy_keys_dict = {"notes.add": {"content"}}

        pred_calls = [
            {"name": "notes.add", "arguments": {"content": "Meeting at 9am tomorrow"}},
            {"name": "alarm.set", "arguments": {"time": "08:00"}}
        ]
        gold_calls = [
            {"name": "notes.add", "arguments": {"content": "meeting at 9am tomorrow"}},
            {"name": "alarm.set", "arguments": {"time": "08:00"}}
        ]

        matches = match_calls(pred_calls, gold_calls, fuzzy_keys_dict)
        self.assertEqual(matches, 2)

    def test_filter_tools(self):
        """Test tool filtering based on installed apps and system tools[cite: 1, 2]."""
        catalog = {
            "smart_home.get_status": {"name": "smart_home.get_status", "kind": "read"},
            "ride.book": {"name": "ride.book", "kind": "write"},
            "sys.ask_missing_info": {"name": "sys.ask_missing_info", "kind": "meta"}
        }
        installed_apps = {
            "smart_home": {"tools": ["smart_home.get_status"]}
        }

        filtered = filter_tools(catalog, installed_apps)
        names = {t["name"] for t in filtered}

        # Should include installed app's tool + sys.ask_missing_info
        self.assertIn("smart_home.get_status", names)
        self.assertIn("sys.ask_missing_info", names)
        self.assertNotIn("ride.book", names)

if __name__ == "__main__":
    unittest.main()

