"""Unit test delle opzioni di retrieval e della ricerca densa su Qdrant."""

import asyncio
from types import SimpleNamespace

from src.core.services import qdrant
from src.core.shared.options import RetrievalOptions


class FakeClient:
    """Registra le chiamate query_points e restituisce punti con payload."""

    def __init__(self, scores=(0.5,)):
        self.calls = []
        self._scores = scores

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        points = [
            SimpleNamespace(id=index, score=score, payload={"page_content": "code", "metadata": {"chunk_id": f"c{index}"}})
            for index, score in enumerate(self._scores, start=1)
        ]
        return SimpleNamespace(points=points)


def test_defaults_enable_every_stage():
    options = RetrievalOptions()
    assert options.k is None and all(value for name, value in vars(options).items() if name != "k")


def test_search_is_one_dense_query():
    client = FakeClient()
    documents = asyncio.run(qdrant.dense_search(client, [0.1], limit=3))
    assert len(client.calls) == 1
    assert client.calls[0]["using"] == "dense" and client.calls[0]["query"] == [0.1]
    assert client.calls[0]["with_payload"] is True and client.calls[0]["limit"] == 3
    assert documents[0].metadata["chunk_id"] == "c1" and documents[0].metadata["_score"] == 0.5


def test_relevance_threshold_is_relative_to_the_best_similarity():
    client = FakeClient(scores=(0.8, 0.5, 0.3))
    documents = asyncio.run(qdrant.dense_search(client, [0.1], limit=3, min_score_ratio=0.5))
    assert [document.metadata["chunk_id"] for document in documents] == ["c1", "c2"]  # soglia 0.4
    assert len(client.calls) == 1


def test_rerank_scores_only_the_given_chunks():
    client = FakeClient()
    asyncio.run(qdrant.rerank_chunk_ids(client, [0.1], ["a", "b"]))
    assert client.calls[0]["limit"] == 2 and len(client.calls[0]["query_filter"].must[0].has_id) == 2
    assert asyncio.run(qdrant.rerank_chunk_ids(client, [0.1], [])) == [] and len(client.calls) == 1
