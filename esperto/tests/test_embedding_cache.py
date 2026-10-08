"""Test della cache persistente degli embedding."""

import numpy as np

from src.core.services.embedding_cache import EmbeddingCache


def test_cache_returns_stored_vectors_and_none_for_unknown_texts(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put(["def f(): pass"], [(np.array([0.5, -0.25], dtype=np.float32), {"12": 0.75})])

    known, unknown = EmbeddingCache(tmp_path / "cache.sqlite").get(["def f(): pass", "def g(): pass"])

    assert unknown is None
    assert known[0].tolist() == [0.5, -0.25]
    assert known[1] == {"12": 0.75}
