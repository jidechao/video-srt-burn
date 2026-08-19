import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import editor


class EditorAppTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        self.video = self.tmp / "demo.mp4"
        self.video.write_bytes(b"fake-video")
        self.transcript = self.tmp / "subtitle-transcript.json"
        self.segments = [
            {"start": 0.0, "end": 2.0, "text": "第一句字幕"},
            {"start": 2.0, "end": 4.0, "text": "第二句字幕"},
        ]
        self.transcript.write_text(
            json.dumps({"segments": self.segments}, ensure_ascii=False), encoding="utf-8"
        )
        self.learning = {
            "status": "ok",
            "learned": [],
            "ignored": [],
            "conflicts": [],
        }

    def _app(self):
        return editor.create_app(
            video_path=self.video,
            transcript_path=self.transcript,
            chapters=[{"index": 1, "title": "开场", "start": 0.0, "end": 60.0}],
            duration=600.0,
            min_progress_duration=180.0,
        )

    def test_get_transcript_serves_segments(self):
        client = self._app().test_client()
        response = client.get("/api/transcript")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["segments"][0]["text"], "第一句字幕")

    def test_manifest_reports_chapters_and_duration(self):
        client = self._app().test_client()
        data = client.get("/manifest").get_json()
        self.assertEqual(data["duration"], 600.0)
        self.assertEqual(data["min_progress_duration"], 180.0)
        self.assertEqual(len(data["chapters"]), 1)
        self.assertEqual(data["languages"][0]["code"], "src")

    def test_index_html_has_no_placeholder_left(self):
        client = self._app().test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"__CACHE_KEY__", response.data)

    def test_post_saves_transcript_and_backs_up_once(self):
        app = self._app()
        client = app.test_client()
        new_segments = [{"start": 0.0, "end": 2.0, "text": "人工修改后的字幕"}]
        with mock.patch.object(
            editor, "learn_manual_edits", return_value=self.learning
        ) as learn:
            response = client.post(
                "/api/transcript", json={"segments": new_segments}
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        learn.assert_called_once()

        saved = json.loads(self.transcript.read_text(encoding="utf-8"))
        self.assertEqual(saved["segments"], new_segments)

        original_path = Path(str(self.transcript) + ".orig.json")
        self.assertTrue(original_path.exists())
        original = json.loads(original_path.read_text(encoding="utf-8"))
        self.assertEqual(original["segments"], self.segments)

        # Second save must not overwrite the first backup.
        with mock.patch.object(editor, "learn_manual_edits", return_value=self.learning):
            client.post("/api/transcript", json={"segments": self.segments})
        original_again = json.loads(original_path.read_text(encoding="utf-8"))
        self.assertEqual(original_again["segments"], self.segments)
        self.assertTrue(app.config["SAVED"]["done"])

    def test_post_rejects_missing_segments(self):
        client = self._app().test_client()
        response = client.post("/api/transcript", json={"nope": 1})
        self.assertEqual(response.status_code, 400)

    def test_learning_failure_still_saves(self):
        app = self._app()
        client = app.test_client()
        with mock.patch.object(
            editor, "learn_manual_edits", side_effect=RuntimeError("model down")
        ):
            response = client.post(
                "/api/transcript", json={"segments": self.segments}
            )
        self.assertEqual(response.status_code, 200)
        learning = response.get_json()["glossary_learning"]
        self.assertEqual(learning["status"], "error")
        self.assertTrue(app.config["SAVED"]["done"])

    def test_relative_paths_still_serve_video(self):
        # Regression: Flask's send_file joins RELATIVE paths onto the app's
        # root_path (the package dir), which broke video serving. create_app
        # must store absolute paths.
        previous_cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(lambda: os.chdir(previous_cwd))
        app = editor.create_app(
            video_path=Path("demo.mp4"),
            transcript_path=Path("subtitle-transcript.json"),
            duration=10.0,
        )
        client = app.test_client()
        response = client.get("/video")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "video/mp4")
        self.assertEqual(response.data, b"fake-video")
        data = client.get("/api/transcript").get_json()
        self.assertEqual(len(data["segments"]), 2)

    def test_load_manifest_resolves_workspace_paths(self):
        manifest = self.tmp / "subtitle-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "video": "demo.mp4",
                    "duration": 123.0,
                    "min_progress_duration": 180.0,
                    "languages": [
                        {"code": "zh", "name": "中文", "transcript": "subtitle-transcript.json", "source": True}
                    ],
                    "chapters_file": "subtitle-chapters.json",
                    "chapters": [{"index": 1, "title": "开场", "start": 0.0, "end": 123.0}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        inputs = editor.load_manifest(manifest)
        self.assertEqual(inputs["video"], self.video)
        self.assertEqual(inputs["transcript"], self.transcript)
        self.assertEqual(inputs["duration"], 123.0)
        self.assertEqual(len(inputs["chapters"]), 1)


if __name__ == "__main__":
    unittest.main()
