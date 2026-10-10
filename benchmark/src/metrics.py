"""Metriche pure di retrieval e statistica (nessun I/O)."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

# Stadio di esperto che porta un chunk tra i candidati, dal più semplice: se più stadi trovano
# lo stesso simbolo vale il primo, così si legge quanto ogni stadio aggiunge alla sola ricerca.
ORIGINS = ("search", "symbol", "same_file", "graph")
# Un simbolo atteso fuori dal contesto: trovato ma escluso dal budget, oppure mai trovato.
LOST = ("budget", "missing")


def symbol_key(source: dict) -> str:
    """Calcola la chiave di un chunk al livello del gold.

    Un metodo (`Classe.metodo`) conta per la sua classe.

    Args:
        source: Fonte o candidato, con `filename` e `symbol_name`.

    Returns:
        La chiave `file::simbolo`.
    """
    return f"{source['filename']}::{str(source.get('symbol_name') or '').split('.')[0]}"


def gold_status(sources: list[dict], dropped: list[dict], relevant: list[dict]) -> list[dict]:
    """Stabilisce dove finisce ogni simbolo atteso.

    Args:
        sources: Fonti entrate nel contesto.
        dropped: Candidati rimasti fuori dal contesto.
        relevant: Simboli attesi, ciascuno con `file`, `symbol` e `hop`.

    Returns:
        Per ogni simbolo atteso la chiave, l'hop e lo stato: lo stadio che lo porta nel contesto,
        `budget` se è stato trovato ma escluso, `missing` se non è mai stato trovato.
    """
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
    """Calcola il reciproco della posizione del primo elemento rilevante.

    Args:
        ranked: Elementi in ordine di rilevanza.
        relevant: Elementi attesi.

    Returns:
        1 diviso la posizione del primo elemento atteso, oppure 0 se non ce n'è nessuno.
    """
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def retrieval_metrics(sources: list[dict], relevant: list[dict]) -> dict[str, float | None]:
    """Calcola le metriche di retrieval di una domanda con gold.

    Args:
        sources: Fonti entrate nel contesto, in ordine di rilevanza.
        relevant: Simboli attesi, ciascuno con `file`, `symbol` e `hop`.

    Returns:
        Le metriche della domanda: `symbol_mrr` (quanto in alto sta il primo simbolo atteso),
        `symbol_context_recall` (quota dei simboli attesi nel contesto, cioè ciò che il modello può
        usare), `hop2_symbol_recall` (lo stesso sui soli simboli raggiungibili seguendo una relazione,
        None se non ce ne sono), `n_sources` e `context_chars`.
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
    """Calcola la media aritmetica.

    Args:
        values: Valori, almeno uno.

    Returns:
        La media dei valori.
    """
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float:
    """Calcola un percentile con interpolazione lineare.

    Args:
        values: Valori, almeno uno.
        p: Percentile richiesto, tra 0 e 100.

    Returns:
        Il valore al percentile `p`.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Stima l'intervallo di confidenza al 95% della media con il bootstrap dei percentili.

    Args:
        values: Valori osservati.
        n_resamples: Numero di ricampionamenti.
        seed: Seme del generatore, per risultati riproducibili.

    Returns:
        Gli estremi inferiore e superiore dell'intervallo.
    """
    rng = random.Random(seed)
    size = len(values)
    means = sorted(mean([values[rng.randrange(size)] for _ in range(size)]) for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Stima la differenza media tra due serie misurate sulle stesse domande (bootstrap appaiato).

    Args:
        a: Valori della prima serie.
        b: Valori della seconda serie, nello stesso ordine.
        n_resamples: Numero di ricampionamenti.
        seed: Seme del generatore, per risultati riproducibili.

    Returns:
        La differenza media `a - b` e gli estremi del suo intervallo di confidenza al 95%.
    """
    low, high = bootstrap_ci([x - y for x, y in zip(a, b)], n_resamples, seed)
    return mean(a) - mean(b), low, high


def share(items: Iterable[str], wanted: Iterable[str]) -> float:
    """Calcola la quota degli elementi che appartengono a un insieme.

    Args:
        items: Elementi da contare.
        wanted: Valori cercati.

    Returns:
        La frazione degli elementi che cadono in `wanted`, 0 se non ci sono elementi.
    """
    items, wanted = list(items), set(wanted)
    return sum(item in wanted for item in items) / len(items) if items else 0.0
