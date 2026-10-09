import pytest

from src.metrics import (
    bootstrap_ci,
    gold_status,
    paired_bootstrap_diff,
    percentile,
    reciprocal_rank,
    retrieval_metrics,
    share,
    symbol_key,
)


def _source(filename: str, symbol: str, origin: str = "search") -> dict:
    return {"filename": filename, "symbol_name": symbol, "origin": origin, "content": "x" * 10}


SOURCES = [
    _source("a.py", "helper"),
    _source("a.py", "Session.send", "same_file"),
    _source("a.py", "Session", "graph"),
    _source("b.py", "get", "symbol"),
]
RELEVANT = [
    {"file": "a.py", "symbol": "Session", "hop": 1},
    {"file": "c.py", "symbol": "cut", "hop": 2},
    {"file": "d.py", "symbol": "lost", "hop": 2},
]


def test_a_method_counts_for_its_class():
    assert symbol_key(_source("a.py", "Session.send")) == "a.py::Session"
    assert symbol_key({"filename": "notes.md", "symbol_name": None}) == "notes.md::"


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b", "z"}) == 0.5
    assert reciprocal_rank(["x"], {"b"}) == 0.0


def test_retrieval_metrics():
    metrics = retrieval_metrics(SOURCES, RELEVANT)
    assert metrics["symbol_mrr"] == 0.5  # a.py::Session e' il secondo simbolo distinto
    assert metrics["symbol_context_recall"] == pytest.approx(1 / 3)
    assert metrics["hop2_symbol_recall"] == 0.0
    assert metrics["n_sources"] == 4 and metrics["context_chars"] == 40
    assert set(metrics) == {"symbol_mrr", "symbol_context_recall", "hop2_symbol_recall", "n_sources", "context_chars"}


def test_retrieval_metrics_without_hop2_or_sources():
    assert retrieval_metrics(SOURCES, RELEVANT[:1])["hop2_symbol_recall"] is None
    empty = retrieval_metrics([], RELEVANT)
    assert empty["symbol_mrr"] == 0.0 and empty["symbol_context_recall"] == 0.0


def test_gold_status_says_where_each_expected_symbol_ends_up():
    dropped = [{"filename": "c.py", "symbol_name": "cut.inner", "origin": "search"}]
    statuses = {item["symbol"]: item["status"] for item in gold_status(SOURCES, dropped, RELEVANT)}
    assert statuses == {
        "a.py::Session": "same_file",  # trovato da due stadi: vale il piu' semplice
        "c.py::cut": "budget",  # tra i candidati, ma fuori dal contesto
        "d.py::lost": "missing",  # mai trovato
    }
    assert [item["hop"] for item in gold_status(SOURCES, dropped, RELEVANT)] == [1, 2, 2]


def test_share():
    assert share(["search", "graph", "missing", "budget"], ["search", "graph"]) == 0.5
    assert share([], ["search"]) == 0.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5


def test_bootstrap_ci_is_reproducible_and_brackets_mean():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    low, high = bootstrap_ci(values, seed=1)
    assert (low, high) == bootstrap_ci(values, seed=1)
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_ci_degenerate_sample():
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_paired_bootstrap_diff_detects_consistent_gain():
    diff, low, high = paired_bootstrap_diff([1.0] * 10, [0.0] * 10)
    assert diff == 1.0 and low == 1.0 and high == 1.0
