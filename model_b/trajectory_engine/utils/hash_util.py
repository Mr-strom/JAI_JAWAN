"""
utils/hash_util.py — SHA-256 helpers for evidence integrity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 65536) -> str:
    """
    Return the hex SHA-256 digest of the file at `path`.
    Reads in chunks so it handles large frames without loading into RAM entirely.
    Returns empty string if the file does not exist (engine keeps running).
    """
    p = Path(path)
    if not p.exists():
        return ""

    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
