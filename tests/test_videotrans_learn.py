import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import learn_glossary


def _segment(start, end, text):
    return {"start": start, "end": end, "text": text}


class CollectManualEditsTests(unittest.TestCase):
    def test_whitespace_only_change_is_not_eligible(self):
        edits = learn_glossary.collect_manual_edits(
            {"segments": [_segment(0.0, 2.0, "你 好 世界")]},
            {"segments": [_segment(0.0, 2.0, "你好 世界")]},
        )
        self.assertTrue(edits)
        self.assertFalse(edits[0]["eligible"])
        self.assertEqual(edits[0]["local_reason"], "仅修改空白")

    def test_punctuation_only_change_is_not_eligible(self):
        edits = learn_glossary.collect_manual_edits(
            {"segments": [_segment(0.0, 2.0, "你好，世界")]},
            {"segments": [_segment(0.0, 2.0, "你好。世界")]},
        )
        self.assertTrue(edits)
        self.assertFalse(edits[0]["eligible"])
        self.assertEqual(edits[0]["local_reason"], "仅修改标点或空白")

    def test_deleted_segment_is_not_eligible(self):
        edits = learn_glossary.collect_manual_edits(
            {"segments": [_segment(0.0, 2.0, "这句话被删除")]},
            {"segments": []},
        )
        self.assertFalse(edits[0]["eligible"])

    def test_minimal_change_extraction(self):
        # The minimal diff is character-level ("r" -> "d"); word-level
        # context is added later by _contextualize_ascii_mapping.
        edits = learn_glossary.collect_manual_edits(
            {"segments": [_segment(0.0, 2.0, "使用 Claude Core 实战")]},
            {"segments": [_segment(0.0, 2.0, "使用 Claude Code 实战")]},
        )
        self.assertTrue(edits[0]["eligible"])
        self.assertEqual(edits[0]["wrong"], "r")
        self.assertEqual(edits[0]["correct"], "d")

    def test_whole_rewrite_is_not_eligible(self):
        edits = learn_glossary.collect_manual_edits(
            {"segments": [_segment(0.0, 2.0, "今天天气不错适合出门散步")]},
            {"segments": [_segment(0.0, 2.0, "完全不同的另一句话出现了")]},
        )
        self.assertFalse(edits[0]["eligible"])


class LearnManualEditsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.glossary = self.tmp / "glossary.json"
        self.report = self.tmp / "manual-edit-review.json"

    def _learn_call(self, items):
        return {"items": items}

    def test_high_confidence_learn_writes_glossary(self):
        before = {"segments": [_segment(0.0, 2.0, "使用 Claude Core 实战")]}
        after = {"segments": [_segment(0.0, 2.0, "使用 Claude Code 实战")]}
        model_items = self._learn_call(
            [
                {
                    "edit_id": 0,
                    "decision": "learn",
                    "wrong": "Claude Core",
                    "correct": "Claude Code",
                    "confidence": 0.99,
                    "reason": "固定产品名",
                }
            ]
        )
        with mock.patch.object(
            learn_glossary,
            "call_qwen_multimodal_json",
            return_value=(model_items, None),
        ):
            report = learn_glossary.learn_manual_edits(
                before, after, report_path=self.report, glossary_path=self.glossary
            )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["learned"]), 1)
        saved = json.loads(self.glossary.read_text(encoding="utf-8"))
        self.assertIn({"wrong": "Claude Core", "correct": "Claude Code"}, saved)
        self.assertTrue(self.report.exists())

    def test_low_confidence_is_ignored(self):
        before = {"segments": [_segment(0.0, 2.0, "使用 Claude Core 实战")]}
        after = {"segments": [_segment(0.0, 2.0, "使用 Claude Code 实战")]}
        model_items = self._learn_call(
            [
                {
                    "edit_id": 0,
                    "decision": "learn",
                    "wrong": "Claude Core",
                    "correct": "Claude Code",
                    "confidence": 0.5,
                    "reason": "不确定",
                }
            ]
        )
        with mock.patch.object(
            learn_glossary,
            "call_qwen_multimodal_json",
            return_value=(model_items, None),
        ):
            report = learn_glossary.learn_manual_edits(
                before, after, report_path=self.report, glossary_path=self.glossary
            )
        self.assertEqual(report["learned"], [])
        self.assertFalse(self.glossary.exists())

    def test_conflict_with_existing_entry_is_recorded(self):
        self.glossary.write_text(
            json.dumps([{"wrong": "Claude Core", "correct": "Claude Pro"}]),
            encoding="utf-8",
        )
        before = {"segments": [_segment(0.0, 2.0, "使用 Claude Core 实战")]}
        after = {"segments": [_segment(0.0, 2.0, "使用 Claude Code 实战")]}
        model_items = self._learn_call(
            [
                {
                    "edit_id": 0,
                    "decision": "learn",
                    "wrong": "Claude Core",
                    "correct": "Claude Code",
                    "confidence": 0.99,
                    "reason": "固定产品名",
                }
            ]
        )
        with mock.patch.object(
            learn_glossary,
            "call_qwen_multimodal_json",
            return_value=(model_items, None),
        ):
            report = learn_glossary.learn_manual_edits(
                before, after, report_path=self.report, glossary_path=self.glossary
            )
        self.assertEqual(len(report["conflicts"]), 1)
        saved = json.loads(self.glossary.read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 1)  # unchanged

    def test_ascii_mapping_gains_neighbor_context(self):
        before = {"segments": [_segment(0.0, 2.0, "use Claude Core now")]}
        after = {"segments": [_segment(0.0, 2.0, "use Claude Code now")]}
        model_items = self._learn_call(
            [
                {
                    "edit_id": 0,
                    "decision": "learn",
                    "wrong": "Core",
                    "correct": "Code",
                    "confidence": 0.99,
                    "reason": "固定产品名",
                }
            ]
        )
        with mock.patch.object(
            learn_glossary,
            "call_qwen_multimodal_json",
            return_value=(model_items, None),
        ):
            report = learn_glossary.learn_manual_edits(
                before, after, report_path=self.report, glossary_path=self.glossary
            )
        self.assertEqual(len(report["learned"]), 1)
        entry = report["learned"][0]
        self.assertEqual(entry["wrong"], "Claude Core")
        self.assertEqual(entry["correct"], "Claude Code")


if __name__ == "__main__":
    unittest.main()
