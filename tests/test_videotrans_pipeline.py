import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.video = self.tmp / "demo.mp4"
        self.video.write_bytes(b"fake-video")
        self.work_dir = self.tmp / "work"

    def test_stage_order_is_fixed(self):
        self.assertEqual(
            [stage.name for stage in pipeline.build_stages()],
            ["transcribe", "review", "prepare", "editor"],
        )

    def _stage_patches(self):
        return {
            name: mock.patch.object(pipeline, f"_{name}_stage")
            for name in ["transcribe", "review", "prepare", "editor"]
        }

    def test_resume_skips_all_completed_stages(self):
        for name in pipeline.build_stages():
            marker = name.marker(self.work_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")

        patches = self._stage_patches()
        with patches["transcribe"] as t, patches["review"] as r, \
                patches["prepare"] as p, patches["editor"] as e:
            report = pipeline.run_pipeline(self.video, work_dir=self.work_dir, resume=True)
        for stage_mock in (t, r, p, e):
            stage_mock.assert_not_called()
        self.assertEqual(
            [stage["status"] for stage in report["stages"]], ["skipped"] * 4
        )
        self.assertTrue(report["saved_in_editor"])
        self.assertTrue(Path(report["final_transcript"]) == self.work_dir / "subtitle-transcript.json")

    def test_fresh_run_executes_all_stages_in_order(self):
        order: list[str] = []

        def editor_stub(work_dir, options):
            order.append("editor")
            (work_dir / "manual-edit-review.json").write_text("{}", encoding="utf-8")

        patches = self._stage_patches()
        with patches["transcribe"] as t, patches["review"] as r, \
                patches["prepare"] as p, patches["editor"] as e:
            t.side_effect = lambda wd, opts: order.append("transcribe")
            r.side_effect = lambda wd, opts: order.append("review")
            p.side_effect = lambda wd, opts: order.append("prepare")
            e.side_effect = editor_stub
            report = pipeline.run_pipeline(self.video, work_dir=self.work_dir)

        self.assertEqual(order, ["transcribe", "review", "prepare", "editor"])
        self.assertEqual(
            [stage["status"] for stage in report["stages"]], ["ran"] * 4
        )
        self.assertTrue(report["saved_in_editor"])
        saved_report = json.loads(
            (self.work_dir / "pipeline-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_report["saved_in_editor"], True)

    def test_stage_failure_aborts_and_records_report(self):
        patches = self._stage_patches()
        with patches["transcribe"] as t, patches["review"] as r, \
                patches["prepare"] as p, patches["editor"] as e:
            t.side_effect = RuntimeError("boom")
            with self.assertRaises(pipeline.PipelineStageError) as ctx:
                pipeline.run_pipeline(self.video, work_dir=self.work_dir)
        self.assertIn("transcribe", str(ctx.exception))
        r.assert_not_called()
        p.assert_not_called()
        e.assert_not_called()
        report = json.loads(
            (self.work_dir / "pipeline-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["stages"][0]["status"], "failed")
        self.assertIn("boom", report["stages"][0]["error"])

    def test_editor_without_save_reports_not_saved(self):
        patches = self._stage_patches()
        with patches["transcribe"], patches["review"], patches["prepare"], \
                patches["editor"] as e:
            e.side_effect = lambda wd, opts: None  # user closed without saving
            report = pipeline.run_pipeline(self.video, work_dir=self.work_dir)
        self.assertFalse(report["saved_in_editor"])
        self.assertEqual(report["stages"][3]["saved"], False)

    def test_missing_video_raises(self):
        with self.assertRaises(FileNotFoundError):
            pipeline.run_pipeline(self.tmp / "absent.mp4")

    def test_default_work_dir_beside_video(self):
        self.assertEqual(
            pipeline.default_work_dir(Path("/clips/demo.mp4")),
            Path("/clips/demo.subtitle-work"),
        )


if __name__ == "__main__":
    unittest.main()
