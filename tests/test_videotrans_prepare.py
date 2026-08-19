import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import prepare
from videotrans.textutil import add_cjk_spacing


def _segments(starts_ends):
    return [
        {"start": start, "end": end, "text": f"第 {index} 段内容，聊一些话题。"}
        for index, (start, end) in enumerate(starts_ends)
    ]


class TextUtilTests(unittest.TestCase):
    def test_add_cjk_spacing(self):
        self.assertEqual(add_cjk_spacing("使用Claude Code做AI实战"), "使用 Claude Code 做 AI 实战")

    def test_add_cjk_spacing_is_idempotent(self):
        once = add_cjk_spacing("版本v2.5发布了")
        self.assertEqual(add_cjk_spacing(once), once)

    def test_display_text_strips_punctuation_and_adds_spacing(self):
        self.assertEqual(
            prepare.display_text("今天，聊聊GPT-5！"),
            "今天聊聊 GPT-5",
        )


class ChapterPlanningTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def _model_chapters(self):
        return {
            "chapters": [
                {"start_id": 0, "title": "开场介绍"},
                {"start_id": 5, "title": "深入原理"},
                {"start_id": 9, "title": "总结展望"},
            ]
        }

    def _long_segments(self):
        starts_ends = []
        for index in range(12):
            start = index * 40.0
            starts_ends.append((start, start + 38.0))
        return _segments(starts_ends)

    def test_chapters_respect_spacing_and_first_at_zero(self):
        with mock.patch.object(
            prepare, "call_qwen_json", return_value=(self._model_chapters(), {"calls": 1})
        ) as model:
            chapters, usage = prepare.plan_chapters(
                self._long_segments(),
                duration=478.0,
                cache_path=self.tmp / "cache" / "chapters-response.json",
            )
        model.assert_called_once()
        self.assertEqual(chapters[0]["start"], 0.0)
        self.assertLessEqual(len(chapters), prepare.DEFAULT_MAX_CHAPTERS)
        for previous, current in zip(chapters, chapters[1:]):
            self.assertGreaterEqual(current["start"] - previous["start"], 75.0)
        self.assertEqual(chapters[-1]["end"], 478.0)

    def test_signature_cache_resumes_without_model_call(self):
        cache_path = self.tmp / "cache" / "chapters-response.json"
        with mock.patch.object(
            prepare, "call_qwen_json", return_value=(self._model_chapters(), {"calls": 1})
        ):
            first, _usage = prepare.plan_chapters(
                self._long_segments(), duration=478.0, cache_path=cache_path
            )
        with mock.patch.object(
            prepare, "call_qwen_json"
        ) as model:
            model.side_effect = AssertionError("model must not be called on resume")
            second, usage = prepare.plan_chapters(
                self._long_segments(),
                duration=478.0,
                cache_path=cache_path,
                resume=True,
            )
        self.assertIsNone(usage)
        self.assertEqual(first, second)

    def test_too_few_broad_chapters_raises(self):
        payload = {"chapters": [{"start_id": 0, "title": "只有一章"}]}
        with mock.patch.object(prepare, "call_qwen_json", return_value=(payload, None)):
            with self.assertRaises(RuntimeError):
                prepare.plan_chapters(self._long_segments(), duration=478.0)


class PrepareSubtitlesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)

    def _write_transcript(self, segments):
        path = self.tmp / "reviewed.json"
        path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
        return path

    def test_short_video_skips_chapters(self):
        transcript = self._write_transcript(
            _segments([(0.0, 4.0), (4.5, 8.0)])
        )
        with mock.patch.object(prepare, "call_qwen_json") as model:
            model.side_effect = AssertionError("no chapter call below threshold")
            report = prepare.prepare_subtitles(
                transcript=transcript,
                output=self.tmp / "subtitle-transcript.json",
                chapters_output=self.tmp / "subtitle-chapters.json",
                progress=True,
                min_progress_duration=180.0,
            )
        self.assertEqual(report["chapters"], [])
        self.assertFalse(report["progress_enabled"])
        chapters_payload = json.loads(
            (self.tmp / "subtitle-chapters.json").read_text(encoding="utf-8")
        )
        self.assertFalse(chapters_payload["enabled"])

    def test_manifest_references_relative_paths(self):
        transcript = self._write_transcript(
            _segments([(0.0, 4.0), (4.5, 8.0)])
        )
        video = self.tmp / "demo.mp4"
        video.write_bytes(b"fake-video")
        with mock.patch.object(prepare, "video_duration", return_value=8.0):
            report = prepare.prepare_subtitles(
                transcript=transcript,
                video=video,
                output=self.tmp / "subtitle-transcript.json",
                chapters_output=self.tmp / "subtitle-chapters.json",
                manifest_output=self.tmp / "subtitle-manifest.json",
                progress=False,
                min_progress_duration=180.0,
            )
        manifest = json.loads(
            (self.tmp / "subtitle-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["video"], "demo.mp4")
        self.assertEqual(manifest["languages"][0]["transcript"], "subtitle-transcript.json")
        self.assertEqual(manifest["chapters_file"], "subtitle-chapters.json")
        self.assertTrue((self.tmp / manifest["video"]).exists())
        self.assertTrue((self.tmp / manifest["languages"][0]["transcript"]).exists())
        self.assertTrue((self.tmp / manifest["chapters_file"]).exists())
        self.assertEqual(report["segments"], 2)

    def test_display_text_applied_to_output_segments(self):
        transcript = self._write_transcript(
            [{"start": 0.0, "end": 3.0, "text": "使用Claude Code做字幕，自动排版。"}]
        )
        prepare.prepare_subtitles(
            transcript=transcript,
            output=self.tmp / "subtitle-transcript.json",
            chapters_output=self.tmp / "subtitle-chapters.json",
            progress=False,
            min_progress_duration=180.0,
        )
        payload = json.loads(
            (self.tmp / "subtitle-transcript.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["segments"][0]["text"], "使用 Claude Code 做字幕自动排版")


if __name__ == "__main__":
    unittest.main()
