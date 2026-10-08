"""Unit test del retriever con client Qdrant ed encoder finti (nessun servizio esterno)."""

import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client.http.models import HasIdCondition

from src.core.rag import retrieval
from src.core.services.qdrant import _point_id
from src.core.shared.options import RetrievalOptions
from src.core.shared.vectors import QueryVectors

# chunk_id -> (file, similarita' densa); l'ordine e' quello della fusione dense+sparse.
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
        return [QueryVectors(dense=[0.1], sparse=None) for _ in texts]

    monkeypatch.setattr(retrieval, "encode_queries", encode)
    return calls


def _retrieve(options=RetrievalOptions(), questions=("sotto-domanda",), related=None, named=(), seeds=None):
    client = FakeClient()
    retriever = retrieval.HybridRetriever(client)

    def related_chunks(chunk_ids, limit):
        if seeds is not None:
            seeds.append(set(chunk_ids))
        return related or {}

    retriever._symbol_graph = SimpleNamespace(related_chunks=related_chunks, chunks_named=lambda text: list(named))
    sources = asyncio.run(retriever.retrieve(list(questions), "domanda", options))
    return client, sources


def test_sources_are_deduplicated_and_ranked_by_a_final_fusion(encoded):
    client, sources = _retrieve()
    # "low" (0.2) e "noise" (0.1) sono sotto la soglia 0.3 * 0.9 della ricerca; "low" rientra
    # come chunk dello stesso file di "top". Senza una classifica comune il primo chunk di ogni
    # ricerca avrebbe lo score massimo: l'ultima chiamata riordina tutti i candidati sulla domanda.
    final = client.calls[-1]
    assert len(final["query_filter"].must[0].has_id) == 3 and final["limit"] == 3
    assert all(prefetch.limit == 3 for prefetch in final["prefetch"])
    assert [source["chunk_id"] for source in sources] == ["low", "top", "mid"]
    assert [source["score"] for source in sources] == [1.0, 0.5, 1 / 3]


def test_relevance_threshold_applies_only_to_the_question_search(encoded):
    _, sources = _retrieve(RetrievalOptions(use_same_file=False))
    assert [source["chunk_id"] for source in sources] == ["top", "mid"]
    assert sources[0]["content"] == "code top"


def test_all_questions_are_encoded_once_with_the_original_first(encoded):
    _retrieve(questions=("Domanda", "prima", "seconda"))
    assert encoded == [["domanda", "prima", "seconda"]]


def test_context_budget_limits_the_sources(encoded, monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_CONTEXT_MAX_SOURCES", 1)
    _, sources = _retrieve()
    assert [source["chunk_id"] for source in sources] == ["low"]


def test_graph_neighbors_skip_the_rerank_and_keep_reserved_slots(encoded, monkeypatch):
    monkeypatch.setattr(retrieval, "RAG_CONTEXT_MAX_SOURCES", 2)
    monkeypatch.setattr(retrieval, "SYMBOL_GRAPH_MAX_NEIGHBORS", 1)
    # "noise" e' sotto la soglia di rilevanza, "top" e' gia' un risultato diretto.
    _, sources = _retrieve(related={"noise": "calls", "top": "called_by:calls"})
    assert [(source["chunk_id"], source["graph_relation"], source["score"]) for source in sources] == [
        ("low", None, 1.0),
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


def test_k_can_be_overridden_per_request(encoded):
    client, _ = _retrieve(RetrievalOptions(k=2, use_same_file=False))
    assert client.calls[0]["limit"] == 2


def test_symbols_named_in_the_question_come_first(encoded):
    # "noise" e' sotto la soglia della ricerca: entra perche' la domanda lo nomina.
    _, sources = _retrieve(named=["noise"])
    assert [source["chunk_id"] for source in sources] == ["noise", "low", "top", "mid"]
    assert sources[0]["score"] == 0.25


def test_symbol_lookup_can_be_disabled(encoded):
    _, sources = _retrieve(RetrievalOptions(use_symbol_lookup=False), named=["noise"])
    assert [source["chunk_id"] for source in sources] == ["low", "top", "mid"]


def test_graph_starts_from_search_results_and_named_symbols_only(encoded):
    seeds = []
    _retrieve(questions=(), named=["noise"], seeds=seeds)
    # "low" entra come chunk dello stesso file di "top": non e' un seme.
    assert seeds == [{"top", "mid", "noise"}]
