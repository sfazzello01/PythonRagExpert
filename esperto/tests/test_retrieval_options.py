import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.core.services import qdrant
from src.core.shared.options import RetrievalOptions
from src.core.shared.vectors import QueryVectors


class FakeClient:
    """Registra le chiamate query_points e restituisce punti con payload."""

    def __init__(self):
        self.calls = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        point = SimpleNamespace(
            id=1, score=0.5, payload={"page_content": "code", "metadata": {"chunk_id": "c1"}}
        )
        return SimpleNamespace(points=[point])


VECTORS = QueryVectors(dense=[0.1], sparse=None)


def _search(options, **kwargs):
    client = FakeClient()
    documents = asyncio.run(qdrant.hybrid_search(client, VECTORS, limit=3, options=options, **kwargs))
    return client, documents


def test_defaults_enable_every_stage():
    options = RetrievalOptions()
    assert options.k is None and all(value for name, value in vars(options).items() if name != "k")


def test_at_least_one_first_stage_is_required():
    with pytest.raises(ValidationError):
        RetrievalOptions(use_dense=False, use_sparse=False)


def test_default_search_is_one_fusion_query():
    client, documents = _search(RetrievalOptions())
    assert len(client.calls) == 1
    assert [p.using for p in client.calls[0]["prefetch"]] == ["dense", "sparse"]
    assert client.calls[0]["with_payload"] is True and client.calls[0]["limit"] == 3
    assert documents[0].metadata["chunk_id"] == "c1" and documents[0].metadata["_score"] == 0.5


def test_relevance_threshold_uses_the_dense_similarity():
    client, documents = _search(RetrievalOptions(), min_score_ratio=0.5)
    assert client.calls[1]["using"] == "dense" and client.calls[1]["query_filter"].must[0].has_id == [1]
    assert len(documents) == 1
    # Senza ricerca densa non c'e' una similarita' su cui filtrare.
    client, _ = _search(RetrievalOptions(use_dense=False), min_score_ratio=0.5)
    assert len(client.calls) == 1


def test_dense_only_has_a_single_prefetch():
    client, _ = _search(RetrievalOptions(use_sparse=False))
    assert [p.using for p in client.calls[0]["prefetch"]] == ["dense"]
