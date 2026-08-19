import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import transcribe


class TempDirTestCase(unittest.TestCase):
    def tmp(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)


class GlossaryTests(TempDirTestCase):
    def test_whitespace_tolerant_pattern_matches_spacing_drift(self):
        pattern = transcribe._glossary_pattern("GPT55")
        self.assertIsNotNone(pattern.search("升级到 GPT 55 啦"))
        self.assertIsNotNone(pattern.search("GPT55"))

    def test_pattern_is_case_insensitive(self):
        pattern = transcribe._glossary_pattern("claude core")
        self.assertIsNotNone(pattern.search("使用 Claude Core 实战"))

    def test_apply_glossary_replaces_and_counts(self):
        glossary = self.tmp() / "glossary.json"
        glossary.write_text(
            json.dumps(
                [{"wrong": "Claude Core", "correct": "Claude Code"},
                 {"wrong": "白练", "correct": "百炼"}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        segments = [
            {"start": 0.0, "end": 2.0, "text": "使用 Claude Core 和白练平台", "words": []},
            {"start": 2.0, "end": 4.0, "text": "没有错词", "words": []},
        ]
        corrected, changed = transcribe.apply_glossary(segments, glossary)
        self.assertEqual(changed, 1)
        self.assertEqual(corrected[0]["text"], "使用 Claude Code 和百炼平台")
        self.assertEqual(corrected[1]["text"], "没有错词")

    def test_apply_glossary_without_file_is_noop(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "白练", "words": []}]
        corrected, changed = transcribe.apply_glossary(segments, None)
        self.assertEqual(changed, 0)
        self.assertEqual(corrected[0]["text"], "白练")


class HotwordVocabularyTests(TempDirTestCase):
    def _hotwords_file(self, tmp: Path) -> Path:
        path = tmp / "hotwords.json"
        path.write_text(
            json.dumps([{"text": "Claude Code", "weight": 4, "lang": "en"}]),
            encoding="utf-8",
        )
        return path

    def test_cache_hit_returns_cached_id_without_network(self):
        tmp = self.tmp()
        hotwords = self._hotwords_file(tmp)
        cache = tmp / "vocabulary-cache.json"
        digest = hashlib.sha256(
            json.dumps(
                json.loads(hotwords.read_text(encoding="utf-8")),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache.write_text(
            json.dumps({"hash": digest, "vocabulary_id": "vocab-123"}), encoding="utf-8"
        )

        with mock.patch.object(transcribe, "load_dashscope_api_key") as key:
            key.side_effect = AssertionError("network path must not be reached")
            vocabulary_id = transcribe.ensure_hotword_vocabulary(hotwords, cache)
        self.assertEqual(vocabulary_id, "vocab-123")

    def test_missing_api_key_degrades_to_none(self):
        tmp = self.tmp()
        hotwords = self._hotwords_file(tmp)
        cache = tmp / "vocabulary-cache.json"
        with mock.patch.object(transcribe, "load_dashscope_api_key", return_value=""):
            vocabulary_id = transcribe.ensure_hotword_vocabulary(hotwords, cache)
        self.assertIsNone(vocabulary_id)

    def test_missing_hotwords_file_returns_none(self):
        tmp = self.tmp()
        self.assertIsNone(transcribe.ensure_hotword_vocabulary(None, tmp / "cache.json"))
        self.assertIsNone(
            transcribe.ensure_hotword_vocabulary(tmp / "absent.json", tmp / "cache.json")
        )


class FillerCleaningTests(TempDirTestCase):
    def test_standalone_filler_words_removed(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "呃我们开始",
                "words": [
                    {"word": "呃", "start": 0.0, "end": 0.4},
                    {"word": "我们开始", "start": 0.4, "end": 2.0},
                ],
            }
        ]
        cleaned, removed = transcribe.clean_fillers(segments)
        self.assertEqual(removed, 1)
        self.assertEqual(cleaned[0]["text"], "我们开始")
        self.assertEqual(cleaned[0]["start"], 0.4)

    def test_text_only_filler_cleanup(self):
        segments = [{"start": 0.0, "end": 2.0, "text": "呃 这个 嗯 东西"}]
        cleaned, removed = transcribe.clean_fillers(segments)
        self.assertEqual(cleaned[0]["text"], "这个 东西")
        self.assertGreaterEqual(removed, 0)


class SplitRuleTests(TempDirTestCase):
    def test_weak_start_detection(self):
        self.assertTrue(transcribe._is_weak_start("的一个原因"))
        self.assertFalse(transcribe._is_weak_start("了解这个工具"))
        self.assertFalse(transcribe._is_weak_start("Claude"))

    def test_rule_split_enforces_hard_caps(self):
        segment = {
            "start": 0.0,
            "end": 12.0,
            "text": "今天我们来聊一聊一个非常长的话题这个话题涉及到很多方面的内容包括第一点第二点第三点以及最后的总结部分",
            "words": [],
        }
        result = transcribe._split_text_proportionally(segment)
        self.assertGreater(len(result), 1)
        for item in result:
            self.assertLessEqual(
                transcribe._visual_len(item["text"]),
                transcribe.MAX_SUBTITLE_CHARS,
            )


class LlmSplitTests(TempDirTestCase):
    def test_llm_output_that_modifies_text_is_rejected(self):
        with mock.patch.object(
            transcribe,
            "call_qwen_text",
            return_value=("第一行改了\n第二行", None),
        ):
            self.assertIsNone(transcribe._llm_split_text("第一行\n第二行"))

    def test_llm_output_matching_text_is_accepted(self):
        with mock.patch.object(
            transcribe,
            "call_qwen_text",
            return_value=("第一行\n第二行", None),
        ):
            self.assertEqual(transcribe._llm_split_text("第一行第二行"), ["第一行", "第二行"])

    def test_alignment_maps_lines_to_word_timestamps(self):
        words = [
            {"word": "我们", "start": 0.0, "end": 0.5},
            {"word": "开始", "start": 0.5, "end": 1.0},
            {"word": "吧", "start": 1.0, "end": 1.2},
        ]
        aligned = transcribe._align_lines_to_words(["我们开始", "吧"], words)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned[0]["start"], 0.0)
        self.assertEqual(aligned[0]["end"], 1.0)
        self.assertEqual(aligned[1]["start"], 1.0)


if __name__ == "__main__":
    unittest.main()
