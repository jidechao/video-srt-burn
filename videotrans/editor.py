#!/usr/bin/env python3
"""Stage 4 — local HTTP server for subtitle preview + editing (single language).

Left pane plays the video with the current-subtitle overlay and chapter
progress; right pane lists subtitles for review, find/replace and deletion.
Saving writes the transcript back, keeps a .orig.json backup, and triggers
glossary learning over the manual edits.

Ported from the oil-subtitle skill's preview_editor.py, single-language only:
the manifest/multi-language/audio-track routes are gone, module globals became
an app factory, and the editor is driven by serve().
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file

from videotrans.learn_glossary import learn_manual_edits


DEFAULT_PORT = 8765
_HTML_PATH = Path(__file__).resolve().parent / "editor.html"


def _editor_port(port: int | None = None) -> int:
    if port is not None:
        return int(port)
    configured = os.environ.get("OILSUBTITLE_EDITOR_PORT") or os.environ.get("PREVIEW_EDITOR_PORT")
    return int(configured or DEFAULT_PORT)


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a subtitle-manifest.json written by prepare.py into editor inputs."""
    manifest_path = Path(manifest_path).resolve()
    workspace = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    language = next(
        (item for item in manifest.get("languages", []) if item.get("source")),
        (manifest.get("languages") or [{}])[0],
    )
    return {
        "video": workspace / manifest["video"],
        "transcript": workspace / language["transcript"],
        "chapters": manifest.get("chapters") or [],
        "duration": float(manifest.get("duration") or 0.0),
        "min_progress_duration": float(manifest.get("min_progress_duration") or 180.0),
    }


def create_app(
    *,
    video_path: Path,
    transcript_path: Path,
    chapters: list[dict[str, Any]] | None = None,
    duration: float = 0.0,
    min_progress_duration: float = 180.0,
) -> Flask:
    """Build the editor Flask app; state lives on app.config for testability."""
    app = Flask(__name__, static_folder=None)
    # Absolute paths: Flask's send_file joins RELATIVE paths onto the app's
    # root_path (the package dir), which breaks video serving.
    app.config.update(
        VIDEO_PATH=str(Path(video_path).resolve()),
        TRANSCRIPT_PATH=str(Path(transcript_path).resolve()),
        CHAPTERS=chapters or [],
        DURATION=float(duration or 0.0),
        MIN_PROGRESS_DURATION=float(min_progress_duration or 180.0),
        SAVED={"done": False},
    )

    def transcript_file() -> Path:
        return Path(app.config["TRANSCRIPT_PATH"])

    def original_file() -> Path:
        return Path(str(transcript_file()) + ".orig.json")

    @app.route("/")
    def index():
        # Include transcript mtime/size so a reused browser tab can never
        # retain an older subtitle payload for the same video.
        path = transcript_file()
        try:
            st = path.stat()
            cache_key = (
                f"videotrans_editor_v1:{app.config['VIDEO_PATH']}:{path}:"
                f"{st.st_mtime_ns}:{st.st_size}"
            )
        except OSError:
            cache_key = f"videotrans_editor_v1:{app.config['VIDEO_PATH']}:{path}"
        html = _HTML_PATH.read_text(encoding="utf-8").replace(
            "__CACHE_KEY__", json.dumps(cache_key)
        )
        return Response(
            html,
            content_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.route("/manifest")
    def manifest():
        return jsonify(
            {
                "video": "/video",
                "duration": app.config["DURATION"],
                "min_progress_duration": app.config["MIN_PROGRESS_DURATION"],
                "chapters": app.config["CHAPTERS"],
                "languages": [
                    {"code": "src", "name": "字幕", "transcript": "src", "source": True}
                ],
            }
        )

    @app.route("/video")
    def video():
        path = Path(app.config["VIDEO_PATH"])
        if not path.exists():
            return "Video not found", 404
        return send_file(path, mimetype="video/mp4")

    @app.route("/api/transcript", methods=["GET"])
    def get_transcript():
        raw = transcript_file().read_text(encoding="utf-8")
        # Guard against stray NaN/Infinity that some writers emit.
        raw = re.sub(r"\bNaN\b", "null", raw)
        raw = re.sub(r"\b-?Infinity\b", "null", raw)
        data = json.loads(raw)
        segs = data.get("segments", data) if isinstance(data, dict) else data
        response = jsonify({"segments": segs})
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.route("/api/transcript", methods=["POST"])
    def post_transcript():
        body = request.get_json()
        segments = body.get("segments")
        if not isinstance(segments, list):
            return jsonify({"ok": False, "error": "segments list required"}), 400

        path = transcript_file()
        original_path = original_file()
        if not original_path.exists():
            original_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(
            json.dumps({"segments": segments}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        learning: dict[str, Any] = {"status": "skipped", "learned_count": 0}
        try:
            report = learn_manual_edits(
                json.loads(original_path.read_text(encoding="utf-8")),
                {"segments": segments},
                report_path=path.parent / "manual-edit-review.json",
            )
            learning = {
                "status": report["status"],
                "learned_count": len(report["learned"]),
                "ignored_count": len(report["ignored"]),
                "conflict_count": len(report["conflicts"]),
                "report": str(path.parent / "manual-edit-review.json"),
            }
        except Exception as exc:
            learning = {"status": "error", "learned_count": 0, "error": str(exc)}
            print(f"[editor] glossary learning failed: {exc}", flush=True)
        app.config["SAVED"]["done"] = True
        return jsonify({"ok": True, "glossary_learning": learning})

    return app


def serve(
    *,
    video_path: Path,
    transcript_path: Path,
    chapters: list[dict[str, Any]] | None = None,
    duration: float = 0.0,
    min_progress_duration: float = 180.0,
    port: int | None = None,
) -> bool:
    """Run the editor until interrupted; returns True when the user saved."""
    video_path = Path(video_path)
    transcript_path = Path(transcript_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    # Snapshot the pre-edit transcript so learning compares against it.
    shutil.copy2(transcript_path, str(transcript_path) + ".orig.json")

    app = create_app(
        video_path=video_path,
        transcript_path=transcript_path,
        chapters=chapters,
        duration=duration,
        min_progress_duration=min_progress_duration,
    )
    resolved_port = _editor_port(port)
    print(f"[editor] Video:     {video_path}")
    print(f"[editor] Transcript: {transcript_path}")
    print(f"[editor] Flask running on http://localhost:{resolved_port}")
    print(f"[editor] 保存并关闭后，回到终端按 Ctrl+C 结束预览。")
    try:
        app.run(host="127.0.0.1", port=resolved_port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        pass
    print("[editor] Exiting.")
    return bool(app.config["SAVED"]["done"])


def serve_from_manifest(manifest_path: Path, *, port: int | None = None) -> bool:
    """Convenience wrapper: drive serve() from a subtitle-manifest.json."""
    inputs = load_manifest(Path(manifest_path))
    return serve(
        video_path=inputs["video"],
        transcript_path=inputs["transcript"],
        chapters=inputs["chapters"],
        duration=inputs["duration"],
        min_progress_duration=inputs["min_progress_duration"],
        port=port,
    )
