#!/usr/bin/env python3
"""Deterministic orchestration of the subtitle pipeline.

Fixed stage order — no model or agent decides what runs next:

  1. transcribe  — ffmpeg audio → Bailian FunAudio ASR (word timestamps)
                   → hotwords + glossary → LLM subtitle splitting
  2. review      — qwen semantic review → frame extraction → vision verify
  3. prepare     — display text → chapters → manifest
  4. editor      — local preview editor (localhost:8765) → save + learning

Each stage records its status in <work-dir>/pipeline-report.json. With
--resume, a stage whose completion marker already exists is skipped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from videotrans import editor, prepare, review, transcribe


DEFAULT_TEXT_MODEL = "qwen3.7-flash"
DEFAULT_VISION_MODEL = "qwen3.7-flash"
DEFAULT_SPLIT_MODEL = "qwen-plus"
DEFAULT_PREPARE_MODEL = "qwen-plus"

WORK_DIR_SUFFIX = ".subtitle-work"

TRANSCRIPT_FILE = "transcript.json"
RAW_ASR_FILE = "bailian_asr.json"
REVIEWED_FILE = "reviewed-transcript.json"
REVIEW_REPORT_FILE = "subtitle-review.json"
REVIEW_FRAMES_DIR = "review-frames"
PREPARED_FILE = "subtitle-transcript.json"
CHAPTERS_FILE = "subtitle-chapters.json"
MANIFEST_FILE = "subtitle-manifest.json"
CACHE_DIR = "cache"
MANUAL_EDIT_REVIEW_FILE = "manual-edit-review.json"
PIPELINE_REPORT_FILE = "pipeline-report.json"


def default_work_dir(video: Path) -> Path:
    video = Path(video)
    return video.parent / (video.stem + WORK_DIR_SUFFIX)


class PipelineStageError(RuntimeError):
    """A stage failed; the message names the stage and the cause."""


class Stage:
    def __init__(
        self,
        name: str,
        marker: Callable[[Path], Path],
        run: Callable[[Path, dict[str, Any]], Any],
    ):
        self.name = name
        self.marker = marker
        self.run = run


def _transcribe_stage(work_dir: Path, options: dict[str, Any]):
    transcribe.transcribe_video(
        Path(options["video"]),
        output_path=work_dir / TRANSCRIPT_FILE,
        language=options["language"],
        raw_output_path=work_dir / RAW_ASR_FILE,
        use_hotwords=options["use_hotwords"],
        apply_glossary_enabled=options["apply_glossary"],
        hotwords_path=options.get("hotwords_path"),
        glossary_path=options.get("glossary_path"),
        split_model=options["split_model"],
    )


def _review_stage(work_dir: Path, options: dict[str, Any]):
    review.review_transcripts(
        video=Path(options["video"]),
        transcript=work_dir / TRANSCRIPT_FILE,
        output=work_dir / REVIEWED_FILE,
        report_path=work_dir / REVIEW_REPORT_FILE,
        frames_dir=work_dir / REVIEW_FRAMES_DIR,
        text_model=options["text_model"],
        vision_model=options["vision_model"],
    )


def _prepare_stage(work_dir: Path, options: dict[str, Any]):
    prepare.prepare_subtitles(
        transcript=work_dir / REVIEWED_FILE,
        video=Path(options["video"]),
        output=work_dir / PREPARED_FILE,
        chapters_output=work_dir / CHAPTERS_FILE,
        manifest_output=work_dir / MANIFEST_FILE,
        work_dir=work_dir / CACHE_DIR,
        model=options["prepare_model"],
        progress=options.get("progress"),
        resume=options["resume"],
    )


def _editor_stage(work_dir: Path, options: dict[str, Any]):
    editor.serve_from_manifest(
        work_dir / MANIFEST_FILE,
        port=options.get("port"),
    )


def build_stages() -> list[Stage]:
    return [
        Stage("transcribe", lambda wd: wd / TRANSCRIPT_FILE, _transcribe_stage),
        Stage("review", lambda wd: wd / REVIEWED_FILE, _review_stage),
        Stage("prepare", lambda wd: wd / PREPARED_FILE, _prepare_stage),
        Stage("editor", lambda wd: wd / MANUAL_EDIT_REVIEW_FILE, _editor_stage),
    ]


def run_pipeline(
    video: Path,
    *,
    work_dir: Path | None = None,
    resume: bool = False,
    language: str | None = "zh",
    use_hotwords: bool = True,
    apply_glossary: bool = True,
    hotwords_path: Path | None = None,
    glossary_path: Path | None = None,
    progress: bool | None = None,
    text_model: str = DEFAULT_TEXT_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
    split_model: str = DEFAULT_SPLIT_MODEL,
    prepare_model: str = DEFAULT_PREPARE_MODEL,
    port: int | None = None,
) -> dict[str, Any]:
    """Run all stages in order; returns the pipeline report dictionary."""
    video = Path(video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    work_dir = Path(work_dir) if work_dir else default_work_dir(video)
    work_dir.mkdir(parents=True, exist_ok=True)

    options: dict[str, Any] = {
        "video": video,
        "language": language,
        "use_hotwords": use_hotwords,
        "apply_glossary": apply_glossary,
        "hotwords_path": Path(hotwords_path) if hotwords_path else None,
        "glossary_path": Path(glossary_path) if glossary_path else None,
        "progress": progress,
        "text_model": text_model,
        "vision_model": vision_model,
        "split_model": split_model,
        "prepare_model": prepare_model,
        "port": port,
        "resume": resume,
    }

    report: dict[str, Any] = {
        "video": str(video.resolve()),
        "work_dir": str(work_dir.resolve()),
        "resume": resume,
        "stages": [],
        "final_transcript": None,
        "saved_in_editor": False,
    }
    report_path = work_dir / PIPELINE_REPORT_FILE

    def write_report():
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    saved_transcript = work_dir / PREPARED_FILE
    for stage in build_stages():
        entry: dict[str, Any] = {"name": stage.name, "status": "pending"}
        report["stages"].append(entry)
        marker = stage.marker(work_dir)
        if resume and marker.exists():
            entry["status"] = "skipped"
            entry["marker"] = str(marker)
            if stage.name == "editor":
                report["saved_in_editor"] = True
                entry["saved"] = True
            if stage.name == "prepare":
                report["final_transcript"] = str(saved_transcript)
            print(f"[pipeline] stage {stage.name}: skipped (marker exists: {marker.name})")
            write_report()
            continue
        started = time.monotonic()
        try:
            stage.run(work_dir, options)
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            entry["seconds"] = round(time.monotonic() - started, 3)
            write_report()
            raise PipelineStageError(f"stage '{stage.name}' failed: {exc}") from exc
        entry["status"] = "ran"
        entry["seconds"] = round(time.monotonic() - started, 3)
        if stage.name == "editor":
            # The editor stage completes when the server exits; the marker is
            # the manual-edit-review.json written on save.
            if marker.exists():
                report["saved_in_editor"] = True
                entry["saved"] = True
            else:
                entry["saved"] = False
        if stage.name == "prepare":
            report["final_transcript"] = str(saved_transcript)
        print(f"[pipeline] stage {stage.name}: done in {entry['seconds']}s")
        write_report()

    if report["saved_in_editor"] and (work_dir / PREPARED_FILE).exists():
        report["final_transcript"] = str(work_dir / PREPARED_FILE)
    write_report()
    print(
        f"[pipeline] finished. Final transcript: {report['final_transcript']} "
        f"(saved in editor: {report['saved_in_editor']})"
    )
    return report
