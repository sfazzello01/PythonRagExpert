"""Unit test del retriever con client Qdrant ed encoder finti (nessun servizio esterno)."""

import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client.http.models import HasIdCondition

from src.core.rag import retrieval
from src.core.services.qdrant import _point_id
from src.core.shared.options import RetrievalOptions
from src.core.shared.vectors import QueryVectors

# chunk_id -> (file, score ColBERT); l'ordine e' quello della fusione dense+sparse.
CHUNKS = {"low": ("a.py", 0.2), "top": ("a.py", 0.9), "mid": ("b.py", 0.6), "noise": ("c.py", 0.1)}


class FakeClient:
    def __init__(self):
        self.calls = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        condition = kwargs["query_filter"].must[0] if kwargs.get("query_filter") else None
        chunk_ids = [
            chunk_id
            for chunk_id, (filename, _) in CHUNKS.items()
            if condition is None
            or (_point_id(chunk_id) in condition.has_id if isinstance(condition, HasIdCondition)
                else filename == condition.match.value)
        ]
        fusion = "prefetch" in kwargs
        points = [
            SimpleNamespace(
                id=_point_id(chunk_id),
                score=1.0 / rank if fusion else CHUNKS[chunk_id][1],
                payload={
                    "page_content": f"code {chunk_id}",
                    "metadata": {"chunk_id": chunk_id, "filename": CHUNKS[chunk_id][0], "symbol_name": chunk_id},
                },
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ]
        return SimpleNamespace(points=points[: kwargs["limit"]])

    async def retrieve(self, **kwargs):
        return [
            SimpleNamespace(payload={"page_content": f"code {chunk_id}", "metadata": {"chunk_id": chunk_id}})
            for chunk_id in CHUNKS
            if _point_id(chunk_id) in kwargs["ids"]
        ]


@pytest.fixture
def encoded(monkeypatch):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return [QueryVectors(dense=[0.1], sparse=None, colbert=[[0.1]]) for _ in texts]

    monkeypatch.setattr(retrieval, "encode_queries", encode)
    return calls


def _retrieve(options=RetrievalOptions(), questions=("sotto-domanda",), related=None):
    client = FakeClient()
    retriever = retrieval.HybridRetriever(client)
    retriever._symbol_graph = SimpleNamespace(related_chunks=lambda chunk_ids, limit: related or {})
    sources = asyncio.run(retriever.retrieve(list(questions), "domanda", options))
    return client, sources


def test_sources_are_deduplicated_and_ranked_by_colbert(encoded):
    _, sources = _retrieve()
    # "noise" (0.1) e' sotto la soglia 0.3 * 0.9 della ricerca; "low" (0.2) anche, ma rientra
    # come chunk dello stesso file di "top" e il rerank finale lo ordina senza scartarlo.
    assert [source["chunk_id"] for source in sources] == ["top", "mid", "low"]
    assert [source["score"] for source in sources] == [0.9, 0.6, 0.2]


def test_relevance_threshold_applies_only_to_the_question_search(encoded):
    _, sources = _retrieve(RetrievalOptions(use_same_file=False))
    assert [source["chunk_id"] for source in sources] == ["top", "mid"]
    assert sources[0]["content"] == "code top"


def test_all_questions_are_encoded_once_with_the_original_first(encoded):
    _retrieve(questions=("Domanda", "prima", "seconda"))
    assert encoded == [["domanda", "prima", "seconda"]]


def test_without_colbert_sources_keep_the_fusion_ranking(encoded):
    client, sources = _retrieve(RetrievalOptions(use_colbert=False, use_same_file=False))
    assert all("using" not in call for call in client.calls)
    assert [source["chunk_id"] for source in sources] == ["low", "top", "mid", "noise"]


def test_context_budget_limits_the_sources(encoded, monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_CONTEXT_MAX_SOURCES", 1)
    _, sources = _retrieve()
    assert [source["chunk_id"] for source in sources] == ["top"]


def test_graph_neighbors_skip_the_rerank_and_keep_reserved_slots(encoded, monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_CONTEXT_MAX_SOURCES", 2)
    monkeypatch.setattr(retrieval, "SYMBOL_GRAPH_MAX_NEIGHBORS", 1)
    # "noise" e' sotto la soglia ColBERT, "top" e' gia' un risultato diretto.
    _, sources = _retrieve(related={"noise": "calls", "top": "called_by:calls"})
    assert [(source["chunk_id"], source["graph_relation"], source["score"]) for source in sources] == [
        ("top", None, 0.9),
        ("noise", "calls", None),
    ]


def test_graph_quota_grows_with_the_number_of_questions(encoded, monkeypatch):
    monkeypatch.setattr(retrieval, "SYMBOL_GRAPH_MAX_NEIGHBORS", 1)
    related = {"noise": "calls", "low": "calls"}
    options = RetrievalOptions(use_same_file=False)
    _, one_question = _retrieve(options, questions=(), related=related)
    _, two_questions = _retrieve(options, related=related)
    assert [source["chunk_id"] for source in one_question if source["graph_relation"]] == ["noise"]
    assert [source["chunk_id"] for source in two_questions if source["graph_relation"]] == ["noise", "low"]
