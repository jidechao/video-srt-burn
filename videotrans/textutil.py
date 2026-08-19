#!/usr/bin/env python3
"""Shared display-text normalization for subtitle preparation."""

from __future__ import annotations

import re


def add_cjk_spacing(text: str) -> str:
    """Add one space at CJK/Latin-or-number boundaries, idempotently."""
    text = re.sub(r"([一-鿿㐀-䶿])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([一-鿿㐀-䶿])", r"\1 \2", text)
    return re.sub(r"[ \t]+", " ", text)
