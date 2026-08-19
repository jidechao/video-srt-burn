import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import review


class ReviewRuleTests(unittest.TestCase):
    def test_versions_are_forced_to_visual_review(self):
        candidates = review.heuristic_visual_candidates(
            [
                {"start": 1, "end": 2, "text": "Grok 4.6 今天发布"},
                {"start": 3, "end": 4, "text": "像 GPT-5.4 一样"},
                {"start": 5, "end": 6, "text": "字幕错成 Grok 46"},
                {"start": 7, "end": 8, "text": "字幕错成 GPT-54"},
            ]
        )
        self.assertEqual(
            [item["original"] for item in candidates],
            ["Grok 4.6", "GPT-5.4", "Grok 46", "GPT-54"],
        )
        self.assertTrue(all(item["decision"] == "visual" for item in candidates))

    def test_commands_and_filenames_are_forced_to_visual_review(self):
        candidates = review.heuristic_visual_candidates(
            [{"start": 1, "end": 2, "text": "运行 --resume 打开 app.py"}]
        )
        self.assertEqual(
            {item["original"] for item in candidates}, {"--resume", "app.py"}
        )

    def test_visual_candidate_overrides_text_only_replacement(self):
        merged = review.merge_candidates(
            [
                {
                    "segment_id": 2,
                    "original": "Grok 4.6",
                    "suggested": "Grok 4.5",
                    "decision": "replace",
                    "confidence": 0.99,
                    "reason": "模型猜测",
                    "source": "text-model",
                },
                {
                    "segment_id": 2,
                    "original": "Grok 4.6",
                    "suggested": "",
                    "decision": "visual",
                    "confidence": 0,
                    "reason": "版本号必须看画面",
                    "source": "rule",
                },
            ]
        )
        self.assertEqual(merged[0]["decision"], "visual")

    def test_safe_replace_requires_one_exact_occurrence(self):
        self.assertEqual(
            review.safe_replace("使用 cloud call", "cloud call", "Claude Code"),
            ("使用 Claude Code", True),
        )
        self.assertEqual(
            review.safe_replace("Grok Grok", "Grok", "Grok 4.6"),
            ("Grok Grok", False),
        )


class ReviewTranscriptsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.video = self.tmp / "demo.mp4"
        self.video.write_bytes(b"fake-video")
        self.transcript = self.tmp / "transcript.json"
        self.segments = [
            {"start": 0.0, "end": 2.0, "text": "使用 Claude Core 做实战"},
            {"start": 2.0, "end": 4.0, "text": "升级到 GPT-5.4 版本"},
        ]
        self.transcript.write_text(json.dumps(self.segments, ensure_ascii=False), encoding="utf-8")

    def test_high_confidence_text_replace_and_vision_keep(self):
        text_result = {
            "items": [
                {
                    "segment_id": 0,
                    "original": "Claude Core",
                    "suggested": "Claude Code",
                    "decision": "replace",
                    "confidence": 0.99,
                    "reason": "固定产品名",
                }
            ]
        }
        vision_result = {
            "decision": "keep",
            "suggested": "",
            "confidence": 0.95,
            "evidence": "画面显示 GPT-5.4",
        }

        def fake_extract_frames(video, segment_id, segment, frames_dir):
            frames_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index in range(3):
                path = frames_dir / f"segment-{segment_id:04d}-{index + 1}.jpg"
                path.write_bytes(b"fake-jpeg")
                paths.append(path)
            return paths

        with mock.patch.object(
            review, "call_qwen_multimodal_json", return_value=(text_result, {"calls": 1})
        ) as text_mock, mock.patch.object(
            review, "call_qwen_vision_json", return_value=(vision_result, {"calls": 1})
        ) as vision_mock, mock.patch.object(
            review, "extract_frames", side_effect=fake_extract_frames
        ):
            report = review.review_transcripts(
                video=self.video,
                transcript=self.transcript,
                output=self.tmp / "reviewed.json",
                report_path=self.tmp / "report.json",
                frames_dir=self.tmp / "frames",
            )

        text_mock.assert_called_once()
        vision_mock.assert_called_once()
        reviewed = json.loads((self.tmp / "reviewed.json").read_text(encoding="utf-8"))
        self.assertEqual(reviewed[0]["text"], "使用 Claude Code 做实战")
        self.assertEqual(reviewed[1]["text"], "升级到 GPT-5.4 版本")
        self.assertEqual(report["summary"]["segments"], 2)
        self.assertEqual(report["summary"]["applied_or_verified"], 2)
        self.assertEqual(report["summary"]["unresolved"], 0)
        self.assertTrue((self.tmp / "report.json").exists())

    def test_unresolved_vision_stays_for_user(self):
        text_result = {"items": []}
        vision_result = {
            "decision": "unresolved",
            "suggested": "",
            "confidence": 0.4,
            "evidence": "画面看不清",
        }

        def fake_extract_frames(video, segment_id, segment, frames_dir):
            return [Path("fake.jpg")]

        with mock.patch.object(
            review, "call_qwen_multimodal_json", return_value=(text_result, None)
        ), mock.patch.object(
            review, "call_qwen_vision_json", return_value=(vision_result, None)
        ), mock.patch.object(
            review, "extract_frames", side_effect=fake_extract_frames
        ):
            report = review.review_transcripts(
                video=self.video,
                transcript=self.transcript,
                output=self.tmp / "reviewed.json",
                report_path=self.tmp / "report.json",
                frames_dir=self.tmp / "frames",
            )

        self.assertEqual(report["summary"]["unresolved"], 1)
        self.assertEqual(
            report["unresolved"][0]["status"], "needs-user-review"
        )


if __name__ == "__main__":
    unittest.main()
