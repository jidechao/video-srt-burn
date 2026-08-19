#!/usr/bin/env python3
"""Command-line entry point: python -m videotrans <video> [options]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from videotrans import __version__
from videotrans.config import save_dashscope_api_key
from videotrans.pipeline import (
    DEFAULT_PREPARE_MODEL,
    DEFAULT_SPLIT_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    PipelineStageError,
    default_work_dir,
    run_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videotrans",
        description=(
            "Deterministic Chinese subtitle pipeline: video -> ASR -> glossary "
            "-> review -> chapters -> local preview editor."
        ),
    )
    parser.add_argument("video", nargs="?", type=Path, help="Local video file (MP4/MOV/…)")
    parser.add_argument("--version", action="version", version=f"videotrans {__version__}")
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Working directory for intermediate files (default: <video>.subtitle-work).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip stages whose output files already exist in the work dir.",
    )
    parser.add_argument(
        "--language", default="zh",
        help="ASR language hint (default: zh; use 'auto' for detection).",
    )
    parser.add_argument(
        "--no-hotwords", action="store_true",
        help="Disable the configured hot-word vocabulary.",
    )
    parser.add_argument("--hotwords", type=Path, default=None, help="Hot-word JSON path override.")
    parser.add_argument(
        "--no-glossary", action="store_true",
        help="Do not apply glossary corrections to the transcript text.",
    )
    parser.add_argument("--glossary", type=Path, default=None, help="Glossary JSON path override.")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None,
        help="Chapter progress on/off (default: config, then on).",
    )
    parser.add_argument("--port", type=int, default=None, help="Preview editor port (default: 8765).")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL, help="Semantic review model.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="Visual verification model.")
    parser.add_argument("--split-model", default=DEFAULT_SPLIT_MODEL, help="Subtitle splitting model.")
    parser.add_argument("--prepare-model", default=DEFAULT_PREPARE_MODEL, help="Chapter planning model.")
    parser.add_argument(
        "--save-api-key", metavar="KEY", default=None,
        help="Save the DashScope API key into the local .env file and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.save_api_key:
        path = save_dashscope_api_key(args.save_api_key)
        print(f"API key saved to {path}")
        return 0

    if not args.video:
        parser.error("a video file is required (or use --save-api-key)")

    language = None if args.language == "auto" else args.language
    try:
        report = run_pipeline(
            args.video,
            work_dir=args.work_dir,
            resume=args.resume,
            language=language,
            use_hotwords=not args.no_hotwords,
            apply_glossary=not args.no_glossary,
            hotwords_path=args.hotwords,
            glossary_path=args.glossary,
            progress=args.progress,
            text_model=args.text_model,
            vision_model=args.vision_model,
            split_model=args.split_model,
            prepare_model=args.prepare_model,
            port=args.port,
        )
    except PipelineStageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    work_dir = Path(report["work_dir"])
    print()
    print("=" * 60)
    print("Pipeline finished")
    print(f"  work dir:        {work_dir}")
    print(f"  final transcript:{report['final_transcript']}")
    print(f"  saved in editor: {report['saved_in_editor']}")
    if args.work_dir is None:
        print(f"  (default work dir was {default_work_dir(args.video)})")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
