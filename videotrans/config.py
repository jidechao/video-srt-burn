#!/usr/bin/env python3
"""Resolve videotrans settings (paths, env vars, DashScope API key).

Configuration is project-local: a .env file next to where you run the
command, plus CLI flags. Hotwords and the glossary are enabled only via the
OIL_SUBTITLE_HOTWORDS / OIL_SUBTITLE_GLOSSARY variables (or the equivalent
--hotwords / --glossary flags); the glossary learning target defaults to a
project-local glossary.json. The DashScope API key is read from
DASHSCOPE_API_KEY in .env (real environment variables take precedence).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PREFERRED_CONFIG = Path.home() / ".config" / "oil-subtitle" / "config.json"
DEFAULT_GLOSSARY = Path("glossary.json")
PREFERRED_VOCABULARY_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "videotrans"
    / "vocabulary-cache.json"
)

_dotenv_loaded_paths: set[str] = set()


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """
    Load KEY=VALUE pairs from a local .env file into os.environ.

    Values already present in the environment are never overridden, so real
    environment variables keep precedence over .env entries. Each file is
    parsed at most once per process. Returns the keys it set.
    """
    path = Path(path) if path else Path.cwd() / ".env"
    if not path.is_file():
        return {}
    resolved = str(path.resolve())
    if resolved in _dotenv_loaded_paths:
        return {}
    _dotenv_loaded_paths.add(resolved)
    loaded: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def config_path() -> Path:
    explicit = env_value("OIL_SUBTITLE_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    return PREFERRED_CONFIG


def load_user_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid videotrans config: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"videotrans config must be a JSON object: {path}")
    return payload


def env_value(name: str) -> str:
    load_env_file()
    return str(os.environ.get(name) or "").strip()


def resolve_progress_enabled(override: bool | None = None) -> bool:
    """Resolve the chapter progress switch; enabled is the safe default."""
    if override is not None:
        return bool(override)
    configured_env = env_value("OIL_SUBTITLE_PROGRESS_ENABLED")
    config = load_user_config()
    subtitle_config = config.get("subtitles") or {}
    if not isinstance(subtitle_config, dict):
        raise RuntimeError("subtitles must be a JSON object in the videotrans config")
    value = (
        configured_env
        if configured_env
        else subtitle_config.get("progress_enabled", True)
    )
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError("subtitles.progress_enabled must be true or false")


def resolve_progress_min_duration(requested: float | None = None) -> float:
    """Resolve the minimum video duration (seconds) for chapter progress."""
    if requested is not None:
        value: Any = requested
    else:
        configured_env = env_value("OIL_SUBTITLE_PROGRESS_MIN_DURATION")
        config = load_user_config()
        subtitle_config = config.get("subtitles") or {}
        if not isinstance(subtitle_config, dict):
            raise RuntimeError(
                "subtitles must be a JSON object in the videotrans config"
            )
        value = configured_env or subtitle_config.get("progress_min_duration_seconds", 180.0)
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Progress minimum duration must be a number of seconds") from exc
    if resolved < 0:
        raise RuntimeError("Progress minimum duration must not be negative")
    return resolved


def optional_user_path(config: dict[str, Any], key: str, env_name: str) -> Path | None:
    value = env_value(env_name) or str(config.get(key) or "").strip()
    return Path(value).expanduser() if value else None


def resolve_glossary_path(override: str | Path | None = None) -> Path:
    """Resolve the glossary: CLI flag, then OIL_SUBTITLE_GLOSSARY (.env/env),
    then the project-local glossary.json."""
    if override:
        return Path(override).expanduser()
    configured = env_value("OIL_SUBTITLE_GLOSSARY")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_GLOSSARY


def resolve_hotwords_path(override: str | Path | None = None) -> Path | None:
    """Resolve the hotwords list path; None (no hot words) unless enabled via
    the CLI flag or OIL_SUBTITLE_HOTWORDS (.env/env)."""
    if override:
        return Path(override).expanduser()
    configured = env_value("OIL_SUBTITLE_HOTWORDS")
    return Path(configured).expanduser() if configured else None


def resolve_vocabulary_cache_path(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    config = load_user_config()
    configured = optional_user_path(config, "vocabulary_cache", "OIL_SUBTITLE_VOCABULARY_CACHE")
    return configured or PREFERRED_VOCABULARY_CACHE


def load_dashscope_api_key(*, required: bool = True) -> str:
    """Read DASHSCOPE_API_KEY from .env (or a real environment variable)."""
    key = env_value("DASHSCOPE_API_KEY")
    if not key and required:
        raise RuntimeError(
            "DashScope API key is not configured. Put DASHSCOPE_API_KEY in the "
            "local .env file, or run `python -m videotrans --save-api-key <KEY>`."
        )
    return key


def save_dashscope_api_key(key: str, path: Path | None = None) -> Path:
    """Upsert DASHSCOPE_API_KEY in the local .env file (other lines kept)."""
    key = str(key or "").strip()
    if not key:
        raise ValueError("DashScope API key must not be empty")
    target = Path(path) if path else Path.cwd() / ".env"
    lines = target.read_text(encoding="utf-8-sig").splitlines() if target.exists() else []
    marker = "DASHSCOPE_API_KEY="
    for index, line in enumerate(lines):
        if line.startswith(marker):
            lines[index] = marker + key
            break
    else:
        lines.append(marker + key)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
