"""Metriche pure: retrieval e statistica (nessun I/O)."""

from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Sequence

# Stadio di esperto che porta un chunk tra i candidati, dal piu' semplice: se piu' stadi trovano
# lo stesso simbolo vale il primo, cosi' si legge quanto ogni stadio aggiunge alla sola ricerca.
ORIGINS = ("search", "symbol", "same_file", "graph")
# Un simbolo atteso fuori dal contesto: trovato ma escluso dal budget, oppure mai trovato.
LOST = ("budget", "missing")


def symbol_key(source: dict) -> str:
    """Chiave di un chunk al livello del gold: un metodo (`Classe.metodo`) conta per la sua classe."""
    return f"{source['filename']}::{str(source.get('symbol_name') or '').split('.')[0]}"


def gold_status(sources: list[dict], dropped: list[dict], relevant: list[dict]) -> list[dict]:
    """Dove finisce ogni simbolo atteso: lo stadio che lo porta nel contesto, `budget` o `missing`."""
    in_context: dict[str, str] = {}
    for source in sources:
        key = symbol_key(source)
        origin = source.get("origin") if source.get("origin") in ORIGINS else "search"
        if key not in in_context or ORIGINS.index(origin) < ORIGINS.index(in_context[key]):
            in_context[key] = origin
    cut = {symbol_key(source) for source in dropped}
    statuses = []
    for item in relevant:
        key = f"{item['file']}::{item['symbol']}"
        status = in_context.get(key) or ("budget" if key in cut else "missing")
        statuses.append({"symbol": key, "hop": item.get("hop", 1), "status": status})
    return statuses


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def retrieval_metrics(sources: list[dict], relevant: list[dict]) -> dict[str, Optional[float]]:
    """Metriche di una domanda con gold; `relevant` = [{file, symbol, hop}].

    - `symbol_mrr`: quanto in alto sta il primo simbolo atteso (ordinamento);
    - `symbol_context_recall`: quota dei simboli attesi nel contesto (cio' che il LLM puo' usare);
    - `hop2_symbol_recall`: lo stesso, sui soli simboli raggiungibili seguendo una relazione.
    """
    ranked = list(dict.fromkeys(symbol_key(source) for source in sources))
    gold = {f"{item['file']}::{item['symbol']}" for item in relevant}
    hop2 = {f"{item['file']}::{item['symbol']}" for item in relevant if item.get("hop", 1) == 2}
    return {
        "symbol_mrr": reciprocal_rank(ranked, gold),
        "symbol_context_recall": len(gold.intersection(ranked)) / len(gold),
        "hop2_symbol_recall": len(hop2.intersection(ranked)) / len(hop2) if hop2 else None,
        # Le configurazioni restituiscono contesti di dimensione diversa: il recall va letto insieme.
        "n_sources": float(len(sources)),
        "context_chars": float(sum(len(source.get("content", "")) for source in sources)),
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


def share(items: Iterable[str], wanted: Iterable[str]) -> float:
    """Quota degli elementi che cadono in `wanted` (0 se non ce ne sono)."""
    items, wanted = list(items), set(wanted)
    return sum(item in wanted for item in items) / len(items) if items else 0.0
