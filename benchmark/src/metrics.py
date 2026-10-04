"""Metriche pure: retrieval, citazioni e statistica (nessun I/O)."""

from __future__ import annotations

import math
import random
import re
from typing import Iterable, Optional, Sequence

KS = (1, 3, 5, 10)


# --- Retrieval (rilevanza binaria su liste ordinate di chiavi) ---

def unique_in_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(item in relevant for item in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(item in relevant for item in top) / len(top) if top else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(pos + 1) for pos, item in enumerate(ranked[:k], start=1) if item in relevant)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def _level_metrics(prefix: str, ranked: list[str], relevant: set[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in KS:
        metrics[f"{prefix}_hit@{k}"] = hit_at_k(ranked, relevant, k)
        metrics[f"{prefix}_recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"{prefix}_precision@{k}"] = precision_at_k(ranked, relevant, k)
    metrics[f"{prefix}_mrr"] = reciprocal_rank(ranked, relevant)
    metrics[f"{prefix}_ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    # Sull'intero contesto restituito: e' cio' che il LLM puo' effettivamente usare.
    metrics[f"{prefix}_context_recall"] = recall_at_k(ranked, relevant, max(len(ranked), 1))
    return metrics


def retrieval_metrics(sources: list[dict], relevant: list[dict]) -> dict[str, Optional[float]]:
    """Metriche file e simbolo per una domanda con gold; `relevant` = [{file, symbol, hop}]."""
    ranked_files = unique_in_order(s["filename"] for s in sources)
    # Il gold e' top-level: un chunk di metodo (`Classe.metodo`) conta per la sua classe.
    ranked_symbols = unique_in_order(f"{s['filename']}::{s['symbol_name'].split('.')[0]}" for s in sources)
    gold_files = {r["file"] for r in relevant}
    gold_symbols = {f"{r['file']}::{r['symbol']}" for r in relevant}
    metrics: dict[str, Optional[float]] = {}
    metrics.update(_level_metrics("file", ranked_files, gold_files))
    metrics.update(_level_metrics("symbol", ranked_symbols, gold_symbols))
    hop2 = {f"{r['file']}::{r['symbol']}" for r in relevant if r.get("hop", 1) == 2}
    metrics["hop2_symbol_recall"] = (
        len(hop2.intersection(ranked_symbols)) / len(hop2) if hop2 else None
    )
    # Dimensione del contesto: le configurazioni ne restituiscono di diverse, e il recall va letto insieme.
    metrics["n_sources"] = float(len(sources))
    metrics["context_chars"] = float(sum(len(s.get("content", "")) for s in sources))
    return metrics


# --- Citazioni nella risposta ---

_PATH_RE = re.compile(r"[\w./-]+\.py\b")


def cited_paths(answer: str) -> set[str]:
    return {match.lstrip("./") for match in _PATH_RE.findall(answer)}


def _matches(cited: str, known: Iterable[str]) -> bool:
    return any(path == cited or path.endswith("/" + cited) for path in known)


def citation_stats(
    answer: str, repo_files: Iterable[str], source_files: Iterable[str]
) -> dict[str, Optional[float]]:
    """Quota di percorsi citati che esistono nel repo / nelle fonti recuperate."""
    cited = cited_paths(answer)
    if not cited:
        return {"citation_exists": None, "citation_in_context": None}
    repo_files, source_files = list(repo_files), list(source_files)
    return {
        "citation_exists": sum(_matches(c, repo_files) for c in cited) / len(cited),
        "citation_in_context": sum(_matches(c, source_files) for c in cited) / len(cited),
    }


# --- Statistica ---

def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Intervallo di confidenza 95% (percentili) della media."""
    rng = random.Random(seed)
    size = len(values)
    means = sorted(mean([values[rng.randrange(size)] for _ in range(size)]) for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Differenza media a-b sulle stesse domande, con CI 95% (bootstrap appaiato)."""
    low, high = bootstrap_ci([x - y for x, y in zip(a, b)], n_resamples, seed)
    return mean(a) - mean(b), low, high


def cohens_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> Optional[float]:
    n = len(labels_a)
    observed = sum(x == y for x, y in zip(labels_a, labels_b)) / n
    p_a, p_b = sum(labels_a) / n, sum(labels_b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return None if expected == 1 else (observed - expected) / (1 - expected)
