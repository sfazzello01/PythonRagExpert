"""Persistent state for skipping unchanged index builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..shared.settings import (
    BASE_DIR,
    DENSE_DIMENSION,
    DENSE_MODEL_NAME,
    EMBEDDING_MAX_LENGTH,
    INDEX_COMPLETE_FILE,
    INDEX_STATE_DIR,
    INDEX_STATE_FILE,
    QDRANT_COLLECTION,
    SYMBOL_GRAPH_FILE,
    SYMBOL_GRAPH_VERSION,
)


def source_fingerprint(files: list[Path]) -> str:
    """Return a stable fingerprint of indexed files and embedding schema settings."""
    digest = hashlib.sha256()
    digest.update(
        f"{DENSE_MODEL_NAME}:{DENSE_DIMENSION}:{EMBEDDING_MAX_LENGTH}".encode("utf-8")
    )
    for path in files:
        try:
            relative_path = path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def is_current(fingerprint: str) -> bool:
    """Return whether the persistent state matches the active index configuration."""
    try:
        state = json.loads(INDEX_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        state.get("collection") == QDRANT_COLLECTION
        and state.get("fingerprint") == fingerprint
        and state.get("symbol_graph_version") == SYMBOL_GRAPH_VERSION
        and SYMBOL_GRAPH_FILE.exists()
    )


def clear() -> None:
    """Remove persistent state before rebuilding the collection."""
    INDEX_STATE_FILE.unlink(missing_ok=True)
    INDEX_COMPLETE_FILE.unlink(missing_ok=True)
    SYMBOL_GRAPH_FILE.unlink(missing_ok=True)


def save(fingerprint: str, source_file_count: int) -> None:
    """Atomically persist the successful index state and completion marker."""
    INDEX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "collection": QDRANT_COLLECTION,
        "fingerprint": fingerprint,
        "source_file_count": source_file_count,
        "symbol_graph_version": SYMBOL_GRAPH_VERSION,
    }
    temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_file.replace(INDEX_STATE_FILE)
    INDEX_COMPLETE_FILE.touch()