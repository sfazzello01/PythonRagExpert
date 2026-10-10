"""Test della cache persistente degli embedding."""

import numpy as np

from src.core.services.embedding_cache import EmbeddingCache


def test_cache_returns_stored_vectors_and_none_for_unknown_texts(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put(["def f(): pass"], [np.array([0.5, -0.25], dtype=np.float32)])

    known, unknown = EmbeddingCache(tmp_path / "cache.sqlite").get(["def f(): pass", "def g(): pass"])

    assert unknown is None
    assert known.tolist() == [0.5, -0.25]


def test_cache_written_by_an_older_version_is_still_readable(tmp_path):
    import sqlite3

    database = sqlite3.connect(tmp_path / "cache.sqlite")  # schema precedente, con una colonna in piu'
    database.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, dense BLOB, sparse TEXT)")
    database.commit()
    database.close()

    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put(["testo"], [np.array([1.0], dtype=np.float32)])
    assert cache.get(["testo"])[0].tolist() == [1.0]
