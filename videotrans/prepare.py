#!/usr/bin/env python3
"""Stage 3 — prepare display-ready Chinese subtitles, broad chapters, and
the preview manifest.

Ported from the oil-subtitle skill's prepare_subtitles.py; the argparse-
coupled main() became a callable prepare_subtitles() with explicit options.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from videotrans.config import (
    resolve_progress_enabled,
    resolve_progress_min_duration,
)
from videotrans.dashscope_client import call_qwen_json
from videotrans.textutil import add_cjk_spacing


DEFAULT_MODEL = "qwen-plus"
CHAPTER_PLANNING_VERSION = 3
DEFAULT_MIN_CHAPTER_DURATION = 75.0
DEFAULT_MAX_CHAPTERS = 6
DISPLAY_PUNCTUATION = re.compile(r"[，。！？；：、,.!?;:…]+")


def log(message: str):
    print(f"[prepare] {message}", flush=True)


def resolve_ffprobe() -> str:
    ffprobe = os.environ.get("FFPROBE") or shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            "ffprobe not found. Install FFmpeg and ensure ffprobe is on PATH."
        )
    return ffprobe


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript has no segments")
    cleaned: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        if not isinstance(raw, dict):
            raise ValueError(f"Transcript segment {index} is not an object")
        text = str(raw.get("text") or "").strip()
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", start))
        if not text or end <= start:
            raise ValueError(f"Transcript segment {index} is empty or has invalid timing")
        cleaned.append(dict(raw))
    return cleaned


def video_duration(path: Path | None, segments: list[dict[str, Any]]) -> float:
    if path:
        if not Path(path).exists():
            raise FileNotFoundError(f"Video does not exist: {path}")
        probe = subprocess.run(
            [
                resolve_ffprobe(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(probe.stdout.strip())
    return max(float(segment["end"]) for segment in segments)


def display_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", DISPLAY_PUNCTUATION.sub("", text)).strip()
    return add_cjk_spacing(normalized)


def signature(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_json(
    *,
    prompt: str,
    model: str,
    timeout: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return call_qwen_json(
        prompt=prompt,
        system="你是视频章节编辑，只返回严格 JSON，不要解释",
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        timeout=timeout,
    )


def chapter_prompt(segments: list[dict[str, Any]], duration: float, max_chapters: int) -> str:
    if duration <= 300:
        min_chapters = 2
        preferred_max_chapters = min(4, max_chapters)
    else:
        min_chapters = min(4, max_chapters)
        preferred_max_chapters = max_chapters
    rows = "\n".join(
        f"[{index} {float(segment['start']):.2f}s] {segment['text']}"
        for index, segment in enumerate(segments)
    )
    return f"""
Plan broad content chapters for this {duration:.1f}-second Mandarin video
Use {min_chapters} to {preferred_max_chapters} chapters total and avoid fragmented topic changes
Each chapter should normally last at least 75 seconds except a short closing
Use a new chapter only when the speaker moves to a different major question
story phase or answer block Do not split every list item into its own chapter

The first chapter must start at segment ID 0 Every later start_id must be an
existing segment ID Title each chapter in concise Chinese using 4 to 10 Chinese
characters and describe the content rather than the editing process

Return strict JSON only
{{"chapters":[{{"start_id":0,"title":"推特意外爆火"}}]}}

TRANSCRIPT
{rows}
""".strip()


def plan_chapters(
    segments: list[dict[str, Any]],
    duration: float,
    *,
    model: str = DEFAULT_MODEL,
    timeout: int = 240,
    max_chapters: int = DEFAULT_MAX_CHAPTERS,
    min_chapter_duration: float = DEFAULT_MIN_CHAPTER_DURATION,
    cache_path: Path | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Plan broad chapters; returns (chapters, model_usage or None)."""
    chapter_signature = signature(
        {
            "version": CHAPTER_PLANNING_VERSION,
            "model": model,
            "duration": duration,
            "segments": [
                [index, item["start"], item["end"], item["text"]]
                for index, item in enumerate(segments)
            ],
            "max_chapters": max_chapters,
            "min_chapter_duration": min_chapter_duration,
        }
    )
    cache_path = Path(cache_path) if cache_path else None
    usage: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    if resume and cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == chapter_signature:
            payload = cached["payload"]
    if payload is None:
        payload, usage = model_json(
            prompt=chapter_prompt(segments, duration, max_chapters),
            model=model,
            timeout=timeout,
            max_tokens=2500,
        )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"signature": chapter_signature, "payload": payload, "usage": usage},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    proposed: list[tuple[int, str]] = []
    for item in payload.get("chapters") or []:
        if not isinstance(item, dict):
            continue
        try:
            start_id = int(item.get("start_id"))
        except (TypeError, ValueError):
            continue
        title = display_text(str(item.get("title") or ""))
        if 0 <= start_id < len(segments) and title:
            proposed.append((start_id, title))
    proposed = sorted(dict(proposed).items())
    if not proposed or proposed[0][0] != 0:
        proposed.insert(0, (0, "内容开场"))

    broad: list[tuple[int, str]] = []
    for start_id, title in proposed:
        start = float(segments[start_id]["start"])
        if broad:
            previous_start = float(segments[broad[-1][0]]["start"])
            if start - previous_start < min_chapter_duration:
                continue
        broad.append((start_id, title))
        if len(broad) >= max_chapters:
            break
    if len(broad) < 2:
        raise RuntimeError("Chapter planner did not produce enough broad chapters")

    chapters: list[dict[str, Any]] = []
    for index, (start_id, title) in enumerate(broad):
        start = 0.0 if index == 0 else float(segments[start_id]["start"])
        end = (
            float(segments[broad[index + 1][0]]["start"])
            if index + 1 < len(broad)
            else duration
        )
        chapters.append(
            {
                "index": index + 1,
                "title": title,
                "start": round(start, 3),
                "end": round(end, 3),
                "start_segment_id": start_id,
            }
        )
    return chapters, usage


def prepare_subtitles(
    *,
    transcript: Path,
    video: Path | None = None,
    output: Path,
    chapters_output: Path,
    manifest_output: Path | None = None,
    work_dir: Path | None = None,
    model: str = DEFAULT_MODEL,
    timeout: int = 240,
    progress: bool | None = None,
    min_progress_duration: float | None = None,
    min_chapter_duration: float = DEFAULT_MIN_CHAPTER_DURATION,
    max_chapters: int = DEFAULT_MAX_CHAPTERS,
    resume: bool = False,
) -> dict[str, Any]:
    """Prepare display subtitles, chapters and manifest; returns a report."""
    progress_enabled = resolve_progress_enabled(progress)
    min_progress = resolve_progress_min_duration(min_progress_duration)

    output = Path(output)
    chapters_output = Path(chapters_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    chapters_output.parent.mkdir(parents=True, exist_ok=True)
    if manifest_output:
        manifest_output = Path(manifest_output)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        if not video:
            raise ValueError("manifest_output requires video")
    if work_dir:
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    segments = load_segments(Path(transcript))
    duration = video_duration(video, segments)
    usages: list[dict[str, Any]] = []

    prepared_segments: list[dict[str, Any]] = []
    for segment in segments:
        prepared = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": display_text(str(segment["text"])),
        }
        prepared_segments.append(prepared)

    chapters: list[dict[str, Any]] = []
    chapter_usage = None
    if progress_enabled and duration > min_progress:
        cache_path = (Path(work_dir) / "chapters-response.json") if work_dir else None
        chapters, chapter_usage = plan_chapters(
            segments,
            duration,
            model=model,
            timeout=timeout,
            max_chapters=max_chapters,
            min_chapter_duration=min_chapter_duration,
            cache_path=cache_path,
            resume=resume,
        )
    if chapter_usage:
        usages.append(chapter_usage)

    subtitle_payload: dict[str, Any] = {
        "schema_version": 1,
        "subtitle_mode": "zh",
        "duration": round(duration, 3),
        "segments": prepared_segments,
        "language": "zh",
    }
    output.write_text(
        json.dumps(subtitle_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chapters_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": bool(chapters),
                "progress_requested": progress_enabled,
                "min_progress_duration": min_progress,
                "duration": round(duration, 3),
                "chapters": chapters,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if manifest_output:
        manifest_dir = manifest_output.parent.resolve()
        video_ref = os.path.relpath(Path(video).resolve(), manifest_dir)
        transcript_ref = os.path.relpath(output.resolve(), manifest_dir)
        chapters_ref = os.path.relpath(chapters_output.resolve(), manifest_dir)
        language = {
            "code": "zh",
            "name": "中文",
            "transcript": transcript_ref,
            "source": True,
        }
        manifest_output.write_text(
            json.dumps(
                {
                    "video": video_ref,
                    "subtitle_mode": "zh",
                    "duration": round(duration, 3),
                    "progress_requested": progress_enabled,
                    "min_progress_duration": min_progress,
                    "languages": [language],
                    "chapters_file": chapters_ref,
                    "chapters": chapters,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "subtitle_mode": "zh",
        "segments": len(prepared_segments),
        "duration": round(duration, 3),
        "progress_enabled": bool(chapters),
        "chapters": chapters,
        "usage": usages,
        "output": str(output),
        "chapters_output": str(chapters_output),
        "manifest_output": str(manifest_output) if manifest_output else None,
    }
    log(
        f"Prepared {len(prepared_segments)} segment(s), duration {duration:.1f}s, "
        f"{len(chapters)} chapter(s)."
    )
    return report
