"""Cache persistente degli embedding dei documenti, indicizzata dal testo codificato."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config import EMBEDDING_CACHE_FILE, EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL_ID


class EmbeddingCache:
    """Archivio SQLite dei vettori già calcolati: i chunk invariati non vengono codificati di nuovo."""

    def __init__(self, path: Path = EMBEDDING_CACHE_FILE):
        """Apre la cache, creandola se non esiste.

        Args:
            path: File SQLite della cache.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dense BLOB)")

    @staticmethod
    def _key(text: str) -> str:
        """Calcola la chiave di un testo nella cache.

        Modello e lunghezza massima fanno parte della chiave: cambiarli non riusa i vettori vecchi.

        Args:
            text: Testo codificato.

        Returns:
            L'hash SHA-256 esadecimale.
        """
        return hashlib.sha256(f"{EMBEDDING_MODEL_ID}:{EMBEDDING_MAX_LENGTH}:{text}".encode("utf-8")).hexdigest()

    def get(self, texts: Sequence[str]) -> list[np.ndarray | None]:
        """Cerca nella cache i vettori dei testi.

        Args:
            texts: Testi da cercare.

        Returns:
            Per ogni testo il vettore in cache, oppure None se non è mai stato codificato.
        """
        select = "SELECT dense FROM embeddings WHERE key = ?"
        rows = [self._db.execute(select, (self._key(text),)).fetchone() for text in texts]
        return [row and np.frombuffer(row[0], dtype=np.float32) for row in rows]

    def put(self, texts: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        """Salva i vettori dei testi appena codificati.

        Args:
            texts: Testi codificati.
            vectors: Vettori corrispondenti, nello stesso ordine.
        """
        rows = [
            (self._key(text), np.asarray(dense, dtype=np.float32).tobytes()) for text, dense in zip(texts, vectors)
        ]
        with self._db:  # un commit per batch: un'indicizzazione interrotta riparte da dove era arrivata
            self._db.executemany("INSERT OR REPLACE INTO embeddings (key, dense) VALUES (?, ?)", rows)
