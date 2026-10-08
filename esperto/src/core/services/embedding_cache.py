"""Persistent cache of document embeddings, keyed by the encoded text."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np

from config import EMBEDDING_CACHE_FILE, EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL_ID

Vectors = tuple[np.ndarray, dict[str, float]]  # dense vector, lexical weights


class EmbeddingCache:
    """SQLite store of the vectors already computed: unchanged chunks are not encoded again."""

    def __init__(self, path: Path = EMBEDDING_CACHE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dense BLOB, sparse TEXT)")

    @staticmethod
    def _key(text: str) -> str:
        # Modello e lunghezza massima fanno parte della chiave: cambiarli non riusa vettori vecchi.
        return hashlib.sha256(f"{EMBEDDING_MODEL_ID}:{EMBEDDING_MAX_LENGTH}:{text}".encode("utf-8")).hexdigest()

    def get(self, texts: Sequence[str]) -> list[Vectors | None]:
        """Return the cached vectors of each text, or None where it has never been encoded."""
        select = "SELECT dense, sparse FROM embeddings WHERE key = ?"
        rows = [self._db.execute(select, (self._key(text),)).fetchone() for text in texts]
        return [row and (np.frombuffer(row[0], dtype=np.float32), json.loads(row[1])) for row in rows]

    def put(self, texts: Sequence[str], vectors: Sequence[Vectors]) -> None:
        """Store the vectors of freshly encoded texts."""
        with self._db:  # un commit per batch: un'indicizzazione interrotta riparte da dove era arrivata
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                [
                    (self._key(text), np.asarray(dense, dtype=np.float32).tobytes(), json.dumps(lexical))
                    for text, (dense, lexical) in zip(texts, vectors)
                ],
            )
