import math

import pytest

from src.metrics import (
    bootstrap_ci,
    citation_stats,
    cohens_kappa,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    unique_in_order,
)

RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d", "z"}


def test_hit_recall_precision_at_k():
    assert hit_at_k(RANKED, RELEVANT, 1) == 0.0
    assert hit_at_k(RANKED, RELEVANT, 2) == 1.0
    assert recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5
    assert precision_at_k([], RELEVANT, 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == 0.5
    assert reciprocal_rank(["x"], RELEVANT) == 0.0


def test_ndcg_matches_hand_computation():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)  # rilevanti in posizione 2 e 4
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 rilevanti, k=10 -> 3 posizioni
    assert ndcg_at_k(RANKED, RELEVANT, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["b"], {"b"}, 5) == 1.0


def test_unique_in_order():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def _source(filename, symbol):
    return {"filename": filename, "symbol_name": symbol}


def test_retrieval_metrics_file_and_symbol_levels():
    sources = [_source("f1.py", "A"), _source("f2.py", "B"), _source("f1.py", "C")]
    relevant = [
        {"file": "f1.py", "symbol": "C", "hop": 1},
        {"file": "f3.py", "symbol": "Z", "hop": 2},
    ]
    metrics = retrieval_metrics(sources, relevant)
    assert metrics["file_hit@1"] == 1.0  # f1.py e' in prima posizione
    assert metrics["file_recall@5"] == 0.5  # f3.py manca
    assert metrics["symbol_hit@1"] == 0.0  # il primo simbolo e' f1.py::A, non rilevante
    assert metrics["symbol_hit@3"] == 1.0
    assert metrics["symbol_mrr"] == pytest.approx(1 / 3)
    assert metrics["symbol_context_recall"] == 0.5
    assert metrics["hop2_symbol_recall"] == 0.0
    assert metrics["n_sources"] == 3.0


def test_retrieval_metrics_without_hop2_is_none():
    metrics = retrieval_metrics([_source("f.py", "A")], [{"file": "f.py", "symbol": "A"}])
    assert metrics["hop2_symbol_recall"] is None


def test_retrieval_metrics_with_no_sources():
    metrics = retrieval_metrics([], [{"file": "f.py", "symbol": "A"}])
    assert metrics["symbol_context_recall"] == 0.0
    assert metrics["symbol_mrr"] == 0.0


def test_citation_stats():
    repo = ["src/requests/utils.py", "src/requests/api.py"]
    sources = ["src/requests/utils.py"]
    answer = "Vedi `src/requests/utils.py`, api.py e fake_module.py."
    stats = citation_stats(answer, repo, sources)
    assert stats["citation_exists"] == pytest.approx(2 / 3)
    assert stats["citation_in_context"] == pytest.approx(1 / 3)


def test_citation_stats_without_citations():
    assert citation_stats("nessun file", [], []) == {"citation_exists": None, "citation_in_context": None}


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
    a = [1.0] * 10
    b = [0.0] * 10
    diff, low, high = paired_bootstrap_diff(a, b)
    assert diff == 1.0 and low == 1.0 and high == 1.0


def test_cohens_kappa():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohens_kappa([True, True], [True, True]) is None
