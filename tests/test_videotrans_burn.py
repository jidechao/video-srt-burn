import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import burn


class BurnTestCase(unittest.TestCase):
    def tmp(self) -> Path:
        import tempfile

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)


class SplitTextTests(BurnTestCase):
    def test_short_text_untouched(self):
        self.assertEqual(burn._split_text("短句", 16), ["短句"])

    def test_long_text_split_within_budget(self):
        text = "今天我们来聊一个非常长的话题涉及到很多方面的内容包括产品定价和渠道运营以及最后的总结部分"
        parts = burn._split_text(text, 12)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(burn._visual_len(part), 12)


class SegmentsToLinesTests(BurnTestCase):
    def test_plain_segment_keeps_text_and_timing(self):
        lines = burn.segments_to_lines(
            [{"start": 0.0, "end": 2.0, "text": "这是一条测试字幕"}], 16
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "这是一条测试字幕")
        self.assertEqual(lines[0]["start"], 0.0)

    def test_noise_lines_filtered(self):
        lines = burn.segments_to_lines(
            [
                {"start": 0.0, "end": 1.0, "text": "呃"},
                {"start": 1.0, "end": 3.0, "text": "正文内容在这里"},
            ],
            16,
        )
        self.assertEqual([line["text"] for line in lines], ["正文内容在这里"])

    def test_word_tokens_drive_timed_lines_when_text_matches(self):
        words = [
            {"word": "我们", "start": 0.0, "end": 0.5},
            {"word": "开始", "start": 0.5, "end": 1.0},
            {"word": "吧", "start": 1.0, "end": 1.2},
        ]
        lines = burn.segments_to_lines(
            [{"start": 0.0, "end": 1.2, "text": "我们开始吧", "words": words}], 16
        )
        self.assertTrue(lines)
        joined = "".join(line["text"] for line in lines)
        self.assertIn("我们", joined)
        for line in lines:
            self.assertGreaterEqual(line["end"], line["start"])

    def test_display_replacements_apply_whitespace_tolerantly(self):
        burn.set_display_replacements([{"wrong": "Claude Core", "correct": "Claude Code"}])
        try:
            lines = burn.segments_to_lines(
                [{"start": 0.0, "end": 2.0, "text": "使用 ClaudeCore 实战"}], 16
            )
            self.assertIn("Claude Code", lines[0]["text"])
        finally:
            burn.set_display_replacements([])


class TimingTests(BurnTestCase):
    def test_normalize_line_timing_fixes_overlap(self):
        lines = burn.normalize_line_timing(
            [
                {"start": 0.0, "end": 2.0, "text": "a"},
                {"start": 1.5, "end": 3.0, "text": "b"},
                {"start": 3.0, "end": 3.0, "text": "c"},
            ]
        )
        for previous, current in zip(lines, lines[1:]):
            self.assertGreaterEqual(current["start"], previous["end"] + 0.02)
        self.assertGreater(lines[-1]["end"], lines[-1]["start"])

    def test_time_converters(self):
        self.assertEqual(burn.seconds_to_ass_time(3661.25), "1:01:01.25")
        self.assertEqual(burn.seconds_to_srt_time(3661.25), "01:01:01,250")
        self.assertEqual(burn.srt_time_to_seconds("01:01:01,250"), 3661.25)


class SrtDraftTests(BurnTestCase):
    def test_write_subtitle_draft_format(self):
        tmp = self.tmp()
        path = tmp / "out.srt"
        burn.write_subtitle_draft(
            [
                {"start": 1.0, "end": 2.5, "text": "第一句"},
                {"start": 3.0, "end": 4.0, "text": "第二句\n换行"},
            ],
            path,
            max_chars=16,
        )
        content = path.read_text(encoding="utf-8")
        self.assertEqual(
            content,
            "1\n00:00:01,000 --> 00:00:02,500\n第一句\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n第二句\n换行\n\n",
        )

    def test_draft_respects_preserve_text(self):
        tmp = self.tmp()
        path = tmp / "out.srt"
        burn.write_subtitle_draft(
            [{"start": 0.0, "end": 1.0, "text": "  keep  me  "}],
            path,
            max_chars=2,  # would re-wrap...
            preserve_text=True,  # ...but preserved verbatim
        )
        self.assertIn("keep  me", path.read_text(encoding="utf-8"))


class ChaptersTests(BurnTestCase):
    def _chapters_file(self, tmp: Path, enabled=True, chapters=None):
        path = tmp / "subtitle-chapters.json"
        path.write_text(
            json.dumps(
                {
                    "enabled": enabled,
                    "min_progress_duration": 180.0,
                    "duration": 400.0,
                    "chapters": chapters if chapters is not None else [
                        {"title": "开场", "start": 0.0, "end": 200.0},
                        {"title": "深入", "start": 200.0, "end": 400.0},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_disabled_or_short_video_has_no_chapters(self):
        tmp = self.tmp()
        self.assertEqual(
            burn.load_progress_chapters(self._chapters_file(tmp), 400.0, enabled=False),
            [],
        )
        self.assertEqual(
            burn.load_progress_chapters(self._chapters_file(tmp), 100.0), []
        )

    def test_valid_chapters_load_and_last_end_pinned(self):
        tmp = self.tmp()
        chapters = burn.load_progress_chapters(self._chapters_file(tmp), 400.0)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["start"], 0.0)
        self.assertEqual(chapters[-1]["end"], 400.0)

    def test_overlap_and_count_are_rejected(self):
        tmp = self.tmp()
        overlapping = [
            {"title": "一", "start": 0.0, "end": 250.0},
            {"title": "二", "start": 200.0, "end": 400.0},
        ]
        with self.assertRaises(ValueError):
            burn.load_progress_chapters(self._chapters_file(tmp, chapters=overlapping), 400.0)
        with self.assertRaises(ValueError):
            burn.load_progress_chapters(
                self._chapters_file(tmp, chapters=[{"title": "唯一", "start": 0.0, "end": 400.0}]),
                400.0,
            )


class AssGenerationTests(BurnTestCase):
    def test_generate_ass_writes_header_and_events(self):
        tmp = self.tmp()
        lines = [{"start": 0.0, "end": 2.0, "text": "你好世界"}]
        path = tmp / "out.ass"
        burn.generate_ass(lines, path, video_width=1920, video_height=1080, max_chars=0)
        content = path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1920", content)
        self.assertIn("PlayResY: 1080", content)
        self.assertIn("CaptionBox", content)
        self.assertIn("你好世界", content)
        # Style definition is always in the header; without chapters there
        # must be no progress Dialogue EVENTS.
        self.assertNotIn("ProgressFill", content)
        self.assertNotIn("ProgressMarker", content)

    def test_progress_events_present_with_chapters(self):
        tmp = self.tmp()
        chapters = [
            {"title": "开场", "start": 0.0, "end": 100.0},
            {"title": "深入", "start": 100.0, "end": 200.0},
        ]
        path = tmp / "out.ass"
        burn.generate_ass(
            [{"start": 0.0, "end": 2.0, "text": "字幕"}],
            path,
            video_width=1280,
            video_height=720,
            chapters=chapters,
            duration=200.0,
        )
        content = path.read_text(encoding="utf-8")
        self.assertIn("ProgressFill", content)
        self.assertIn("ProgressMarker", content)
        self.assertIn("开场", content)
        self.assertIn("深入", content)


class FilterGraphTests(BurnTestCase):
    def test_progress_graph_uses_gradient_not_solid_canvas(self):
        graph, out = burn.build_progress_filter_graph(
            "ass='x.ass'", progress_overlay_height=54
        )
        self.assertEqual(out, "[video_out]")
        self.assertIn("geq=r='47'", graph)
        self.assertIn("overlay=0:main_h-overlay_h", graph)
        self.assertIn("ass='x.ass'", graph)

    def test_beauty_graph_validates_inputs(self):
        with self.assertRaises(ValueError):
            burn.build_beauty_filter_graph(
                "ass='x.ass'", 100, 100, (10, 10, 200, 10)
            )
        with self.assertRaises(ValueError):
            burn.build_beauty_filter_graph(
                "ass='x.ass'", 100, 100, (0, 0, 50, 50),
                smoothing_strength=0, brighten_strength=0,
            )

    def test_beauty_graph_structure(self):
        graph, out = burn.build_beauty_filter_graph(
            "ass='x.ass'", 1920, 1080, (100, 100, 300, 300),
            progress_overlay_height=54,
        )
        self.assertIn("crop=300:300:100:100", graph)
        self.assertIn("bilateral=sigmaS=2.5", graph)
        self.assertIn("overlay=100:100", graph)
        self.assertTrue(graph.index("overlay=100:100") < graph.index("geq=r='47'"))
        self.assertTrue(graph.index("geq=r='47'") < graph.rindex("ass='x.ass'"))


class CameraRegionTests(BurnTestCase):
    def test_persistent_cluster_becomes_even_square(self):
        detections = []
        for sample in range(8):
            detections.append({
                "sample": sample, "x": 0.60, "y": 0.10,
                "width": 0.20, "height": 0.30, "confidence": 0.9,
            })
        region = burn.derive_camera_region(detections, 1920, 1080, 8)
        self.assertIsNotNone(region)
        x, y, side, side2 = region
        self.assertEqual(side, side2)
        self.assertEqual(side % 2, 0)
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x + side, 1920)

    def test_no_persistent_face_returns_none(self):
        detections = [
            {"sample": 0, "x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
        ]
        self.assertIsNone(burn.derive_camera_region(detections, 1920, 1080, 8))


class SrtTests(BurnTestCase):
    def test_read_srt_roundtrip(self):
        tmp = self.tmp()
        path = tmp / "reviewed.srt"
        path.write_text(
            "1\n00:00:01,000 --> 00:00:02,500\n第一句\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n第二句\n",
            encoding="utf-8",
        )
        lines = burn.read_srt_lines(path)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["start"], 1.0)
        self.assertEqual(lines[1]["text"], "第二句")

    def test_overlapping_srt_rejected(self):
        tmp = self.tmp()
        path = tmp / "bad.srt"
        path.write_text(
            "1\n00:00:02,000 --> 00:00:04,000\na\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nb\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            burn.read_srt_lines(path)


class LayoutHelperTests(BurnTestCase):
    def test_safe_max_chars_positive(self):
        self.assertGreater(burn._safe_max_chars_for_video(1920, 1080), 4)
        self.assertGreater(burn._safe_max_chars_for_video(720, 1280), 4)

    def test_font_size_scales_proportionally(self):
        # Regression: a 44px floor made 270p subtitles cover ~16% of frame
        # height. Small videos must keep their proportional share (~6%).
        small_font, _ = burn._subtitle_style_metrics(480, 270)
        self.assertLessEqual(small_font / 270, 0.10)
        self.assertGreaterEqual(small_font, 20)

        hd_font, _ = burn._subtitle_style_metrics(1920, 1080)
        self.assertEqual(hd_font, 65)

        tiny_font, _ = burn._subtitle_style_metrics(320, 180)
        self.assertEqual(tiny_font, 20)

    def test_landscape_default_wider_than_portrait(self):
        landscape = burn._resolve_effective_max_chars(0, 1920, 1080, False)
        portrait = burn._resolve_effective_max_chars(0, 720, 1280, False)
        self.assertGreater(landscape, portrait)


if __name__ == "__main__":
    unittest.main()
