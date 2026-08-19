#!/usr/bin/env python3
"""Stage 1 — extract audio, run Bailian FunAudio ASR, apply hotwords/glossary,
and split the transcript into subtitle lines.

Ported from the oil-subtitle skill's bailian_transcribe.py; stage paths are
explicit parameters instead of module globals so the stage is testable.

Output format:
  [{"start": 0.0, "end": 1.23, "text": "...", "words": [...]}]
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from videotrans.config import (
    load_dashscope_api_key,
    resolve_glossary_path,
    resolve_hotwords_path,
    resolve_vocabulary_cache_path,
)
from videotrans.dashscope_client import call_qwen_text, transcribe_audio_file


ASR_MODEL = "fun-asr"
VOCABULARY_PREFIX = "videotrans"
MAX_SUBTITLE_CHARS = 24
MAX_SUBTITLE_SECONDS = 4.2
TARGET_SUBTITLE_SECONDS = 2.4
SOFT_PUNCT = "，,、；;：:"
HARD_PUNCT = "。！？!?"
FILLER_CHARS = set("呃嗯啊额诶唉哦噢")
FILLER_LATIN = {"em", "emm", "um", "uh", "er", "erm"}
DISPLAY_PUNCT = SOFT_PUNCT + HARD_PUNCT + "…."

DEFAULT_SPLIT_MODEL = "qwen-plus"
_SPLIT_WORKERS = 6


def log(message: str):
    print(f"[transcribe] {message}", flush=True)


def resolve_ffmpeg() -> str:
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install FFmpeg and ensure it is on PATH, or set "
            "the FFMPEG environment variable."
        )
    return ffmpeg


def _run(cmd: list[str]):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd[:3])}\n{detail}")


def _extract_audio(input_path: Path, output_path: Path):
    _run([
        resolve_ffmpeg(),
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ])


def ensure_hotword_vocabulary(
    hotwords_path: Path | None, cache_path: Path
) -> str | None:
    """
    Create or reuse a Bailian hot-word vocabulary from the configured file.

    Hot words steer ASR toward recurring proper nouns (Claude, 百炼…) at
    recognition time — errors that glossary text replacement can only
    partially patch afterwards. The vocabulary ID is cached outside the
    repository keyed by a hash of the hot-word list, so the remote
    vocabulary is only created/updated when the list changes. Any failure
    degrades to plain recognition — never block transcription.
    """
    if hotwords_path is None or not hotwords_path.exists():
        return None
    try:
        hotwords = json.loads(hotwords_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"WARNING: could not read hot words ({exc}); continuing without them.")
        return None
    if not isinstance(hotwords, list) or not hotwords:
        return None

    digest = hashlib.sha256(
        json.dumps(hotwords, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    cache: dict[str, Any] = {}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    if cache.get("hash") == digest and cache.get("vocabulary_id"):
        return cache["vocabulary_id"]

    api_key = load_dashscope_api_key(required=False)
    if not api_key:
        log("WARNING: no DashScope API key found; continuing without hot words.")
        return None

    try:
        import dashscope
        from dashscope.audio.asr import VocabularyService

        dashscope.api_key = api_key
        service = VocabularyService(api_key=api_key)
        vocabulary_id = cache.get("vocabulary_id")
        if vocabulary_id:
            try:
                service.update_vocabulary(vocabulary_id, hotwords)
            except Exception:
                vocabulary_id = None  # stale/deleted remotely — recreate below
        if not vocabulary_id:
            vocabulary_id = service.create_vocabulary(
                target_model=ASR_MODEL,
                prefix=VOCABULARY_PREFIX,
                vocabulary=hotwords,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"hash": digest, "vocabulary_id": vocabulary_id}),
            encoding="utf-8",
        )
        log(f"Hot-word vocabulary ready ({len(hotwords)} terms): {vocabulary_id}")
        return vocabulary_id
    except Exception as exc:
        log(f"WARNING: hot-word vocabulary unavailable ({exc}); continuing without it.")
        return None


def _glossary_pattern(wrong: str) -> re.Pattern:
    """
    Case-insensitive AND whitespace-tolerant pattern for a glossary entry.
    Spacing drifts at every stage (ASR tokens, LLM echo, CJK/Latin spacing),
    so "GPT55" must also match "GPT 55" and "cloud call" must match
    "cloudcall" — otherwise entries silently stop matching.
    """
    parts = [re.escape(ch) for ch in wrong if not ch.isspace()]
    return re.compile(r"\s*".join(parts), re.IGNORECASE)


def load_glossary_replacements(glossary_path: Path | None) -> list[tuple[re.Pattern, str]]:
    """Load configured glossary as case-insensitive replacement patterns."""
    if glossary_path is None or not glossary_path.exists():
        return []
    try:
        entries = json.loads(glossary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    replacements = []
    for entry in entries if isinstance(entries, list) else []:
        wrong = (entry.get("wrong") or "").strip()
        correct = entry.get("correct")
        if wrong and isinstance(correct, str):
            replacements.append((_glossary_pattern(wrong), correct))
    return replacements


def apply_glossary(
    segments: list[dict[str, Any]], glossary_path: Path | None
) -> tuple[list[dict[str, Any]], int]:
    """
    Apply recurring glossary corrections to segment display text after ASR.

    Corrections are applied at transcription time so they are visible in the
    preview editor and the review stage can focus on new, unknown errors.
    Word-level tokens keep the raw ASR text; segment "text" is the display
    source of truth.
    """
    replacements = load_glossary_replacements(glossary_path)
    if not replacements:
        return segments, 0
    changed = 0
    corrected = []
    for segment in segments:
        text = segment.get("text") or ""
        new_text = text
        for pattern, correct in replacements:
            new_text = pattern.sub(correct, new_text)
        if new_text != text:
            changed += 1
        corrected.append({**segment, "text": new_text})
    return corrected, changed


def _ms(value: Any) -> float:
    return round(float(value or 0) / 1000, 3)


def _word_text(word: dict[str, Any]) -> str:
    text = (word.get("text") or word.get("word") or "").strip()
    punctuation = (word.get("punctuation") or "").strip()
    return (text + punctuation).strip()


def _visual_len(text: str) -> float:
    width = 0.0
    for char in text:
        if "一" <= char <= "鿿" or "㐀" <= char <= "䶿":
            width += 1.0
        elif char == " ":
            width += 0.5
        else:
            width += 0.55
    return width


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[\s，,、；;：:。！？!?…+\-_/()（）\"'`]+", "", text).lower()


def _concat_words(words: list[dict[str, Any]]) -> str:
    return "".join((word.get("word") or "").strip() for word in words).strip()


def _filler_core(text: str) -> str:
    return re.sub(rf"[\s{re.escape(DISPLAY_PUNCT)}]+", "", text).lower()


def _is_standalone_filler(text: str) -> bool:
    core = _filler_core(text)
    if not core:
        return False
    if core in FILLER_LATIN:
        return True
    return all(char in FILLER_CHARS for char in core)


def _clean_filler_text(text: str) -> str:
    filler = r"(?:呃+|嗯+|啊+|额+|诶+|唉+|哦+|噢+|em+|um+|uh+|er+)"
    boundary = rf"(^|[\s{re.escape(DISPLAY_PUNCT)}])"
    cleaned = re.sub(
        rf"{boundary}{filler}(?=$|[\s{re.escape(DISPLAY_PUNCT)}])",
        lambda match: match.group(1),
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"([，,、；;：:]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"([。！？!?]){2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ，,、；;：:。！？!?…")


def _strip_display_punctuation(text: str) -> str:
    """Remove punctuation used for timing but not wanted in visible subtitles."""
    return re.sub(rf"[{re.escape(DISPLAY_PUNCT)}]", "", text)


def _clean_display_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = _strip_display_punctuation(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_segment_display_text(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = _clean_display_text(segment.get("text") or "")
        if text:
            cleaned.append({**segment, "text": text})
    return cleaned


def clean_fillers(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    cleaned_segments: list[dict[str, Any]] = []
    removed = 0
    for segment in segments:
        words = segment.get("words") or []
        if words:
            kept_words = []
            for word in words:
                if _is_standalone_filler(word.get("word") or ""):
                    removed += 1
                    continue
                kept_words.append(word)
            if not kept_words:
                continue
            cleaned_segments.append({
                **segment,
                "start": round(float(kept_words[0]["start"]), 3),
                "end": round(float(kept_words[-1]["end"]), 3),
                "text": _concat_words(kept_words),
                "words": kept_words,
            })
            continue

        cleaned_text = _clean_filler_text(segment.get("text") or "")
        if cleaned_text != (segment.get("text") or "").strip():
            removed += 1
        if cleaned_text:
            cleaned_segments.append({**segment, "text": cleaned_text})
    return cleaned_segments, removed


def _line_duration(words: list[dict[str, Any]]) -> float:
    if not words:
        return 0.0
    return max(0.0, float(words[-1]["end"]) - float(words[0]["start"]))


_WEAK_START_CHARS = {"的", "了", "着", "过", "们", "吗", "呢", "吧", "啊"}
# Content words that legitimately begin with a particle character — starting a
# subtitle with these is fine ("了解一下", "过程中"…).
_WEAK_START_WHITELIST = (
    "的确", "了解", "了不起", "着重", "着急", "着手", "着眼",
    "过程", "过去", "过后", "过来", "过于", "过年", "过度", "过滤",
)


def _is_weak_start(text: str) -> bool:
    """
    True when a subtitle must not START with this token: a grammatical particle
    gluing it to the previous phrase ("的一个…", "的原因…"). ASR tokens often
    attach 的/了 to the following word, so check the first character, not just
    single-character tokens.
    """
    if not text or text[0] not in _WEAK_START_CHARS:
        return False
    return not text.startswith(_WEAK_START_WHITELIST)


def _is_bare_cjk(text: str) -> bool:
    return len(text) == 1 and bool(re.fullmatch(r"[一-鿿]", text))


def _boundary_score(words: list[dict[str, Any]], start: int, end: int) -> float:
    chunk = words[start:end + 1]
    text = _concat_words(chunk)
    duration = _line_duration(chunk)
    length = _visual_len(text)
    tail = text[-1] if text else ""
    next_word = words[end + 1] if end + 1 < len(words) else None
    end_word_text = (words[end].get("word") or "").strip()
    next_text = (next_word.get("word") or "").strip() if next_word else ""
    gap = max(0.0, float(next_word["start"]) - float(words[end]["end"])) if next_word else 0.0

    score = 0.0
    score -= abs(length - MAX_SUBTITLE_CHARS * 0.78) * 1.4
    score -= abs(duration - TARGET_SUBTITLE_SECONDS) * 3.0

    if tail in HARD_PUNCT:
        score += 38
    if gap >= 0.35:
        score += 28
    elif gap >= 0.22:
        score += 18
    elif tail in SOFT_PUNCT:
        score += 12

    if length < MAX_SUBTITLE_CHARS * 0.38 and next_word:
        score -= 34
    if duration < 1.0 and next_word:
        score -= 34
    if length > MAX_SUBTITLE_CHARS:
        score -= 90 + (length - MAX_SUBTITLE_CHARS) * 14
    if duration > MAX_SUBTITLE_SECONDS:
        score -= (duration - MAX_SUBTITLE_SECONDS) * 30
    if _is_weak_start(next_text):
        score -= 80
    if next_word and _is_bare_cjk(end_word_text) and tail not in SOFT_PUNCT + HARD_PUNCT:
        score -= 28
    # Never break between Latin/digit characters without punctuation or a real
    # pause: ASR tokenizes English words and phrases into arbitrary pieces
    # ("FIVEF" + "IVE"), and a subtitle boundary there splits a word across
    # two subtitles. Only a forced overflow break may do that.
    # "." counts as part of the run so version numbers ("GPT5" + "." + "5")
    # are not split either.
    if (
        next_word is not None
        and re.fullmatch(r"[A-Za-z0-9.]", (end_word_text or " ")[-1])
        and re.fullmatch(r"[A-Za-z0-9.]", (next_text or " ")[0])
        and tail not in SOFT_PUNCT + HARD_PUNCT
        and gap < 0.22
    ):
        score -= 250
    if next_word is None:
        score += 20
    return score


def _split_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    while start < len(words):
        best_end = start
        best_score = float("-inf")
        for end in range(start, len(words)):
            chunk = words[start:end + 1]
            text = _concat_words(chunk)
            duration = _line_duration(chunk)
            length = _visual_len(text)
            if end > start and (length > MAX_SUBTITLE_CHARS * 1.08 or duration > MAX_SUBTITLE_SECONDS):
                break
            score = _boundary_score(words, start, end)
            if score > best_score:
                best_score = score
                best_end = end
        chunks.append(words[start:best_end + 1])
        start = best_end + 1
    return chunks


def _split_text_chunks(text: str, max_chars: float = MAX_SUBTITLE_CHARS) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text or _visual_len(text) <= max_chars:
        return [text] if text else []

    parts: list[str] = []

    def add_piece(piece: str):
        piece = piece.strip()
        if not piece:
            return
        if _visual_len(piece) <= max_chars:
            parts.append(piece)
            return
        soft_chunks = [p.strip() for p in re.split(r"(?<=[，,、；;：:])\s*", piece) if p.strip()]
        if len(soft_chunks) > 1:
            for chunk in soft_chunks:
                add_piece(chunk)
            return
        current = ""
        for char in piece:
            if current and _visual_len(current + char) > max_chars:
                parts.append(current.strip())
                current = char
            else:
                current += char
        if current.strip():
            parts.append(current.strip())

    hard_chunks = [p.strip() for p in re.split(r"(?<=[。！？!?])\s*", text) if p.strip()]
    for chunk in hard_chunks:
        add_piece(chunk)
    return parts or [text]


def _split_text_proportionally(segment: dict[str, Any]) -> list[dict[str, Any]]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(0.01, end - start)
    chunks = _split_text_chunks(segment["text"])
    min_parts = max(1, math.ceil(duration / MAX_SUBTITLE_SECONDS))
    if len(chunks) < min_parts:
        target_width = max(10.0, min(float(MAX_SUBTITLE_CHARS), _visual_len(segment["text"]) / min_parts))
        chunks = _split_text_chunks(segment["text"], target_width)
    if len(chunks) <= 1:
        return [segment]
    weights = [max(1.0, _visual_len(chunk)) for chunk in chunks]
    total = sum(weights)
    result = []
    cursor = start
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        next_end = end if index == len(chunks) - 1 else cursor + duration * (weight / total)
        result.append({
            "start": round(cursor, 3),
            "end": round(next_end, 3),
            "text": chunk,
            "words": [],
        })
        cursor = next_end
    return result


def _append_short_segment(target: list[dict[str, Any]], segment: dict[str, Any]):
    duration = float(segment["end"]) - float(segment["start"])
    if duration > MAX_SUBTITLE_SECONDS or _visual_len(segment["text"]) > MAX_SUBTITLE_CHARS:
        target.extend(_split_text_proportionally({**segment, "words": []}))
    else:
        target.append(segment)


def _shorten_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shortened: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words") or []
        text = (segment.get("text") or "").strip()
        needs_split = (
            _visual_len(text) > MAX_SUBTITLE_CHARS
            or float(segment["end"]) - float(segment["start"]) > MAX_SUBTITLE_SECONDS
        )
        if not needs_split:
            shortened.append(segment)
            continue

        if words and _normalize_for_match(_concat_words(words)) == _normalize_for_match(text):
            for chunk_words in _split_words(words):
                _append_short_segment(shortened, {
                    "start": round(float(chunk_words[0]["start"]), 3),
                    "end": round(float(chunk_words[-1]["end"]), 3),
                    "text": _concat_words(chunk_words),
                    "words": chunk_words,
                })
        else:
            for item in _split_text_proportionally(segment):
                _append_short_segment(shortened, item)

    shortened.sort(key=lambda item: (item["start"], item["end"]))
    return shortened


_SPLIT_SYSTEM_PROMPT = """你是字幕断句助手。用户给你一句视频口播的文字，你把它断成适合做视频字幕的短行。

规则：
1. 只允许插入换行符来断行，绝对不许增加、删除、修改、调换任何字符（包括标点）。
2. 用尽量少的行数完成切分，每行不超过 24 个汉字宽度（中文字符算 1，英文字母、数字算半个）。
3. 在自然的语气停顿处断开，每行是语义完整的短语，相邻行长度尽量均衡。
4. 不要把英文产品名、版本号（如 Claude Code、GPT 5.5、Gemini 3.5 Flash）拆到两行。
5. 行首不要是「的、了、着、吗、呢、吧、啊」这类粘在前一个短语上的虚词。
6. 定语和它修饰的中心语放在同一行；「XX的」和后面的名词不要拆开。
7. 直接输出断好行的文字，不要任何解释、编号或多余内容。"""


# Normalization for validating/aligning LLM-split lines: LLMs tend to "fix"
# punctuation even when told not to, and display punctuation is stripped from
# subtitles anyway — so comparison and word alignment ignore whitespace and
# punctuation entirely. Word characters are still matched exactly.
_ALIGN_NORM_RE = re.compile(
    rf"[\s{re.escape(DISPLAY_PUNCT)}\-—–―‐~·«»“”‘’\"'()（）《》〈〉「」『』【】\[\]{{}}]+"
)


def _align_norm(text: str) -> str:
    return _ALIGN_NORM_RE.sub("", text)


def _llm_split_text(full_text: str, model: str = DEFAULT_SPLIT_MODEL) -> list[str] | None:
    """
    Ask an LLM to insert subtitle line breaks into the transcript text.

    The LLM only chooses break points — the reassembled output must be
    character-identical to the input (whitespace ignored), otherwise the
    result is rejected and the caller falls back to rule-based splitting.
    """
    try:
        content, _usage = call_qwen_text(
            prompt=full_text,
            system=_SPLIT_SYSTEM_PROMPT,
            model=model,
            max_tokens=8192,
            temperature=0.1,
            timeout=180,
        )
    except Exception as exc:
        log(f"WARNING: LLM split call failed: {exc}")
        return None

    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        log("WARNING: LLM split returned no lines.")
        return None

    if _align_norm("".join(lines)) != _align_norm(full_text):
        log("WARNING: LLM split modified the text; rejecting and falling back to rules.")
        return None
    return lines


def _align_lines_to_words(
    lines: list[str], words: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """
    Map LLM-split text lines back to ASR word timestamps.

    Words are concatenated in order; each line consumes its own characters
    (whitespace and punctuation ignored — see _align_norm) and takes start/end
    from the first/last word it touches. Any bookkeeping mismatch aborts the
    whole alignment.
    """
    flat = ""
    char_owner: list[int] = []
    for idx, word in enumerate(words):
        text = _align_norm(word.get("word") or "")
        flat += text
        char_owner.extend([idx] * len(text))

    segments = []
    pos = 0
    for line in lines:
        stripped = _align_norm(line)
        if not stripped:
            continue
        start_pos, end_pos = pos, pos + len(stripped)
        if end_pos > len(flat) or flat[start_pos:end_pos] != stripped:
            return None
        owners = char_owner[start_pos:end_pos]
        first_word, last_word = words[owners[0]], words[owners[-1]]
        segments.append({
            "start": round(float(first_word["start"]), 3),
            "end": round(float(last_word["end"]), 3),
            "text": line,
            "words": [words[i] for i in sorted(set(owners))],
        })
        pos = end_pos
    if pos != len(flat):
        return None
    return segments


def _is_fragment(segment: dict[str, Any]) -> bool:
    min_chars = max(8.0, MAX_SUBTITLE_CHARS * 0.38)
    duration = float(segment["end"]) - float(segment["start"])
    return (
        _visual_len(_clean_display_text(segment.get("text") or "")) < min_chars
        or duration < 0.95
    )


def _can_join(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Never join across a sentence boundary — a fragment glued to the wrong
    # sentence reads worse than a short line.
    left_text = (left.get("text") or "").rstrip()
    if left_text and left_text[-1] in HARD_PUNCT:
        return False
    combined_len = _visual_len(_clean_display_text(left_text + (right.get("text") or "")))
    combined_dur = float(right["end"]) - float(left["start"])
    gap = float(right["start"]) - float(left["end"])
    return combined_len <= MAX_SUBTITLE_CHARS and combined_dur <= MAX_SUBTITLE_SECONDS and gap <= 0.45


def _join(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        **left,
        "text": (left.get("text") or "") + (right.get("text") or ""),
        "end": right["end"],
        "words": (left.get("words") or []) + (right.get("words") or []),
    }


def _merge_short_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Absorb fragments too short to read comfortably into a neighbor.
    Backward merge first (into the previous line), forward merge as fallback
    (a fragment that starts a new sentence belongs to the line after it).
    """
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and _is_fragment(seg) and _can_join(merged[-1], seg):
            merged[-1] = _join(merged[-1], seg)
        else:
            merged.append(dict(seg))

    result: list[dict[str, Any]] = []
    i = 0
    while i < len(merged):
        seg = merged[i]
        if i + 1 < len(merged) and _is_fragment(seg) and _can_join(seg, merged[i + 1]):
            result.append(_join(seg, merged[i + 1]))
            i += 2
        else:
            result.append(seg)
            i += 1
    return result


def _needs_split(segment: dict[str, Any]) -> bool:
    text = segment.get("text") or ""
    duration = float(segment.get("end", 0)) - float(segment.get("start", 0))
    return (
        _visual_len(_clean_display_text(text)) > MAX_SUBTITLE_CHARS
        or duration > MAX_SUBTITLE_SECONDS
    )


def _llm_split_segment(segment: dict[str, Any], model: str) -> list[dict[str, Any]] | None:
    """
    Split ONE ASR sentence with LLM-chosen line breaks and re-time the lines
    from the sentence's word timestamps. Returns None on any failure so the
    caller can fall back to rule-based splitting for just this sentence.
    """
    words = segment.get("words") or []
    if not words:
        return None
    lines = _llm_split_text(segment.get("text") or "", model=model)
    if not lines:
        return None
    # One retry for a line the model left over-wide; a short line is a small
    # input it splits reliably.
    fixed: list[str] = []
    for line in lines:
        if _visual_len(_clean_display_text(line)) > MAX_SUBTITLE_CHARS:
            sub = _llm_split_text(line, model=model)
            if sub and len(sub) > 1:
                fixed.extend(sub)
                continue
        fixed.append(line)
    return _align_lines_to_words(fixed, words)


def llm_segment(
    segments: list[dict[str, Any]], model: str = DEFAULT_SPLIT_MODEL
) -> list[dict[str, Any]] | None:
    """
    Split over-long ASR sentences with LLM-chosen line breaks, sentence by
    sentence, in parallel.

    Per-sentence inputs keep the LLM reliable (echoing a whole transcript
    makes it miscount widths and "fix" wording) and contain failures: one bad
    response degrades one sentence to rule-based splitting instead of the
    whole video. Sentences that already fit a subtitle line skip the LLM.
    """
    todo = [i for i, seg in enumerate(segments) if _needs_split(seg)]
    if not todo:
        return segments

    results: dict[int, list[dict[str, Any]] | None] = {}
    with ThreadPoolExecutor(max_workers=_SPLIT_WORKERS) as pool:
        futures = {i: pool.submit(_llm_split_segment, segments[i], model) for i in todo}
        for i, fut in futures.items():
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None

    failed = sum(1 for i in todo if not results.get(i))
    output: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        split = results.get(i)
        if split:
            output.extend(split)
        else:
            output.append(seg)  # unchanged; _shorten_segments rule-splits it
    log(f"LLM segmentation: {len(todo) - failed}/{len(todo)} long sentence(s) split"
        + (f", {failed} fell back to rules" if failed else "") + ".")
    return output


def _convert_bailian_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    transcripts = data.get("transcripts") or []
    if not transcripts:
        return []

    segments: list[dict[str, Any]] = []
    for sentence in transcripts[0].get("sentences") or []:
        text = (sentence.get("text") or "").strip()
        if not text:
            continue

        words = []
        for word in sentence.get("words") or []:
            clean_text = _word_text(word)
            if not clean_text:
                continue
            begin = word.get("begin_time", word.get("start"))
            end = word.get("end_time", word.get("end"))
            if begin is None or end is None:
                continue
            words.append({
                "word": clean_text,
                "start": _ms(begin),
                "end": _ms(end),
            })

        begin = sentence.get("begin_time", sentence.get("start"))
        end = sentence.get("end_time", sentence.get("end"))
        segments.append({
            "start": _ms(begin),
            "end": _ms(end),
            "text": text,
            "words": words,
        })

    segments.sort(key=lambda item: (item["start"], item["end"]))
    return segments


def transcribe_video(
    input_path: Path,
    *,
    output_path: Path | None = None,
    language: str | None = "zh",
    raw_output_path: Path | None = None,
    clean_fillers_enabled: bool = True,
    use_hotwords: bool = True,
    apply_glossary_enabled: bool = True,
    hotwords_path: Path | None = None,
    glossary_path: Path | None = None,
    vocabulary_cache_path: Path | None = None,
    split_model: str = DEFAULT_SPLIT_MODEL,
) -> list[dict[str, Any]]:
    """Run the full transcription stage for one video file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    if hotwords_path is None:
        hotwords_path = resolve_hotwords_path()
    if glossary_path is None:
        glossary_path = resolve_glossary_path()
    if vocabulary_cache_path is None:
        vocabulary_cache_path = resolve_vocabulary_cache_path()

    vocabulary_id: str | None = None
    if use_hotwords:
        vocabulary_id = ensure_hotword_vocabulary(hotwords_path, vocabulary_cache_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio_for_bailian.wav"
        log("Extracting 16 kHz mono audio...")
        _extract_audio(input_path, audio_path)

        lang_display = language or "auto"
        log(f"Transcribing with Bailian FunAudio ASR (language={lang_display}"
            + (", hot words on" if vocabulary_id else "") + ")...")
        raw = transcribe_audio_file(
            audio_path,
            model=ASR_MODEL,
            language=language,
            vocabulary_id=vocabulary_id,
        )
        if raw_output_path:
            raw_output_path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    raw_segments = _convert_bailian_result(raw)
    cleaned_segments, removed_fillers = (
        clean_fillers(raw_segments) if clean_fillers_enabled else (raw_segments, 0)
    )

    log(f"Splitting subtitles with {split_model}...")
    llm_segments = llm_segment(cleaned_segments, model=split_model)
    if llm_segments is None:
        log("Falling back to rule-based subtitle splitting.")
    base_segments = llm_segments if llm_segments else cleaned_segments
    # _shorten_segments enforces the hard width/duration caps; on LLM output it
    # only touches lines that exceed them. The merge pass then absorbs
    # too-short fragments.
    segments = _clean_segment_display_text(
        _merge_short_segments(_shorten_segments(base_segments))
    )

    glossary_hits = 0
    if apply_glossary_enabled:
        segments, glossary_hits = apply_glossary(segments, glossary_path)
    if output_path:
        output_path.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"Saved transcript: {output_path}")
    log(
        f"Transcribed {len(raw_segments)} ASR sentence(s), removed {removed_fillers} filler token(s), "
        f"applied glossary to {glossary_hits} segment(s), "
        f"split into {len(segments)} subtitle segment(s), "
        f"{sum(len(s.get('words', [])) for s in segments)} words."
    )
    return segments
