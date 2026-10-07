"""Aggregazione dei risultati di un run in summary.json/csv e report.md."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from .metrics import bootstrap_ci, cohens_kappa, mean, paired_bootstrap_diff, percentile

RETRIEVAL_COLUMNS = [
    ("file_hit@5", "File Hit@5"),
    ("file_recall@5", "File Recall@5"),
    ("symbol_hit@5", "Symbol Hit@5"),
    ("symbol_recall@5", "Symbol Recall@5"),
    ("symbol_mrr", "Symbol MRR"),
    ("symbol_ndcg@10", "Symbol nDCG@10"),
    ("symbol_context_recall", "Symbol ctx recall"),
    ("hop2_symbol_recall", "Hop-2 recall"),
    ("n_sources", "N fonti"),
    ("context_chars", "Caratteri contesto"),
]
DELTA_METRICS = ["symbol_mrr", "symbol_recall@5", "symbol_context_recall", "hop2_symbol_recall"]
GENERATION_COLUMNS = [
    ("correctness", "Correttezza"),
    ("faithfulness", "Fedelta'"),
    ("citation_exists", "Citazioni esistenti"),
    ("citation_in_context", "Citazioni nel contesto"),
    ("hallucination", "Hallucination (senza risposta)"),
    ("false_refusal", "Falsi rifiuti"),
    ("answer_error", "Errori"),
    ("judged", "Giudicate"),
]
LATENCY_COLUMNS = [("retrieval_s", "Retrieval (s)"), ("ttft_s", "TTFT (s)"), ("total_s", "Totale (s)")]


def _load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load(path: Path) -> list[dict]:
    rows = _load_lines(path)
    # Dopo una ripresa una domanda puo' comparire due volte: vale l'ultimo tentativo.
    return list({(row["config"], row["id"]): row for row in rows}.values())


def _configs(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(row["config"] for row in rows))


def _values(rows: list[dict], config: str, metric: str, category: Optional[str] = None) -> list[float]:
    return [
        row["metrics"][metric]
        for row in rows
        if row["config"] == config
        and (category is None or row.get("category") == category)
        and row.get("metrics", {}).get(metric) is not None
    ]


def _by_id(rows: list[dict], config: str, metric: str) -> dict[str, float]:
    return {
        row["id"]: row["metrics"][metric]
        for row in rows
        if row["config"] == config and row.get("metrics", {}).get(metric) is not None
    }


def _stat(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci_lo": None, "ci_hi": None, "p50": None, "p95": None}
    low, high = bootstrap_ci(values) if len(values) > 1 else (values[0], values[0])
    return {
        "n": len(values),
        "mean": mean(values),
        "ci_lo": low,
        "ci_hi": high,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def _cell(stat: dict) -> str:
    if stat["n"] == 0:
        return "n/d"
    return f"{stat['mean']:.2f} [{stat['ci_lo']:.2f}, {stat['ci_hi']:.2f}]"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _summary_entries(stage: str, rows: list[dict], metrics: list[str]) -> list[dict]:
    entries = []
    for config in _configs(rows):
        for metric in metrics:
            entries.append({"stage": stage, "config": config, "category": "all", "metric": metric,
                            **_stat(_values(rows, config, metric))})
    return entries


def _metric_names(rows: list[dict]) -> list[str]:
    return sorted({name for row in rows for name in row.get("metrics", {})})


def _metric_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    return _table(
        ["Config", "N"] + [label for _, label in columns],
        [
            [c, str(sum(row["config"] == c for row in rows))] + [_cell(_stat(_values(rows, c, m))) for m, _ in columns]
            for c in _configs(rows)
        ],
    )


def _delta_table(rows: list[dict]) -> str:
    out = []
    for config in [c for c in _configs(rows) if c != "full"]:
        cells = []
        for metric in DELTA_METRICS:
            a, b = _by_id(rows, config, metric), _by_id(rows, "full", metric)
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                cells.append("n/d")
                continue
            diff, low, high = paired_bootstrap_diff([a[i] for i in common], [b[i] for i in common])
            cells.append(f"{diff:+.2f} [{low:+.2f}, {high:+.2f}]{'*' if low > 0 or high < 0 else ''}")
        out.append([config] + cells)
    return _table(["Config vs full"] + DELTA_METRICS, out)


def _category_table(rows: list[dict], metric: str) -> str:
    categories = sorted({r["category"] for r in rows if r.get("metrics")})
    return _table(
        ["Config"] + categories,
        [[c] + [_cell(_stat(_values(rows, c, metric, cat))) for cat in categories] for c in _configs(rows)],
    )


def _error_note(rows: list[dict]) -> list[str]:
    """Avviso sulle richieste fallite, che non entrano nelle medie."""
    failed = sum(bool(row.get("error")) for row in rows)
    return [f"**Attenzione**: {failed} richieste fallite su {len(rows)}, escluse dalle medie."] if failed else []


def _latency_table(rows: list[dict]) -> str:
    def cell(config: str, metric: str) -> str:
        stat = _stat(_values(rows, config, metric))
        return "n/d" if stat["n"] == 0 else f"{stat['p50']:.2f} / {stat['p95']:.2f}"

    return _table(
        ["Config"] + [f"{label} p50 / p95" for _, label in LATENCY_COLUMNS],
        [[c] + [cell(c, m) for m, _ in LATENCY_COLUMNS] for c in _configs(rows)],
    )


def _apply_judgments(run_dir: Path, run_id: str, generation: list[dict]) -> None:
    """Unisce i verdetti di `judgments.jsonl` alle risposte e ne ricava le metriche."""
    judgments = {row["key"]: row for row in _load_lines(run_dir / "judgments.jsonl")}
    facts = {row["key"]: len(row["key_facts"]) for row in _load_lines(run_dir / "to_judge.jsonl")}
    for row in generation:
        if row.get("error"):
            continue
        key = hashlib.sha1(f"{run_id}:{row['config']}:{row['id']}".encode("utf-8")).hexdigest()[:12]
        judgment = judgments.get(key)
        answerable = row["category"] != "unanswerable"
        valid = judgment is not None and len(judgment.get("fact_verdicts") or []) == facts.get(key, -1)
        row["metrics"]["judged"] = float(valid)
        if not valid:
            continue
        row["fact_verdicts"], row["abstains"] = judgment["fact_verdicts"], judgment["abstains"]
        # hallucination: risposta data a una domanda che non ha risposta; false_refusal: il contrario.
        if answerable:
            row["metrics"]["correctness"] = sum(judgment["fact_verdicts"]) / len(judgment["fact_verdicts"])
            row["metrics"]["false_refusal"] = float(judgment["abstains"])
        else:
            row["metrics"]["hallucination"] = float(not judgment["abstains"])
        if judgment.get("faithful") is not None:
            row["metrics"]["faithfulness"] = float(judgment["faithful"])


def _judge_agreement(generation: list[dict], labels_path: Path) -> str:
    if not labels_path.exists():
        return "Nessun file `human_labels.json`: validazione del giudice non eseguita."
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = {(r["config"], r["id"]): r for r in generation}
    pairs: dict[str, tuple[list[bool], list[bool]]] = {"fatti": ([], []), "fedelta'": ([], []), "astensione": ([], [])}
    for label in labels:
        row = rows.get((label["config"], label["id"]))
        if row is None:
            continue
        if row.get("fact_verdicts") and "fact_verdicts" in label:
            pairs["fatti"][0].extend(label["fact_verdicts"])
            pairs["fatti"][1].extend(row["fact_verdicts"])
        if "faithful" in label and "faithfulness" in row.get("metrics", {}):
            pairs["fedelta'"][0].append(label["faithful"])
            pairs["fedelta'"][1].append(bool(row["metrics"]["faithfulness"]))
        if "abstains" in label and row.get("abstains") is not None:
            pairs["astensione"][0].append(label["abstains"])
            pairs["astensione"][1].append(row["abstains"])
    table = []
    for name, (human, judge) in pairs.items():
        if not human:
            table.append([name, "0", "n/d", "n/d"])
            continue
        kappa = cohens_kappa(human, judge)
        agreement = sum(h == j for h, j in zip(human, judge)) / len(human)
        table.append([name, str(len(human)), f"{agreement:.2f}", "n/d" if kappa is None else f"{kappa:.2f}"])
    return _table(["Verdetto", "N", "Accordo", "Cohen's kappa"], table)


def _write_csv(path: Path, entries: list[dict]) -> None:
    fields = ["stage", "config", "category", "metric", "n", "mean", "ci_lo", "ci_hi", "p50", "p95"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def write_report(run_dir: Path, labels_path: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    retrieval_rows = _load(run_dir / "retrieval.jsonl")
    retrieval = [r for r in retrieval_rows if "metrics" in r]
    generation = _load(run_dir / "generation.jsonl")
    _apply_judgments(run_dir, manifest["run_id"], generation)
    judged = sum(row.get("metrics", {}).get("judged", 0) for row in generation)
    latency = [r for r in generation if "total_s" in r.get("metrics", {})]  # tempi misurati in generazione

    entries = (
        _summary_entries("retrieval", retrieval, _metric_names(retrieval))
        + _summary_entries("generation", generation, _metric_names(generation))
    )
    for category in sorted({r["category"] for r in retrieval}):
        for config in _configs(retrieval):
            for metric in _metric_names(retrieval):
                stat = _stat(_values(retrieval, config, metric, category))
                if stat["n"]:
                    entries.append({"stage": "retrieval", "config": config, "category": category, "metric": metric, **stat})
    (run_dir / "summary.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    _write_csv(run_dir / "summary.csv", entries)

    dataset, job = manifest["dataset"], manifest.get("indexing_job") or {}
    categories = ", ".join(f"{name}: {count}" for name, count in dataset["questions_by_category"].items())
    header = "\n".join([
        f"- Dataset: `{dataset['name']}` ({dataset['repo_url']} @ `{dataset['ref']}`, commit `{dataset['commit'][:10]}`)",
        f"- Gold set: {categories} (sha256 `{dataset['questions_sha256'][:12]}`)",
        f"- Campione: {dataset.get('sampled_questions', 'n/d')} domande"
        f" (per categoria: {manifest['spec'].get('per_category') or 'tutte'}, seed {manifest['spec'].get('seed', 0)})",
        f"- Modello esperto: `{manifest['rag_params']['LLM_MODEL']}`; giudice esterno: {int(judged)} risposte giudicate"
        " (`to_judge.jsonl` -> `POST /runs/{id}/judgments`)",
        f"- Indicizzazione: {job.get('indexing_seconds', 'n/d')} s, {manifest['esperto_health'].get('points_count', 'n/d')} punti",
        "- Celle: media [IC 95% bootstrap sulle domande]; `*` = differenza appaiata con IC che esclude 0.",
    ])
    sections = [f"# Benchmark {manifest['run_id']}", header]
    if retrieval:
        sections += [
            "## Retrieval", *_error_note(retrieval_rows), _metric_table(retrieval, RETRIEVAL_COLUMNS),
            "### Ablation: differenza rispetto a `full`", _delta_table(retrieval),
            "### Symbol Recall@5 per categoria", _category_table(retrieval, "symbol_recall@5"),
        ]
    if generation:
        sections += [
            "## Generazione", _metric_table(generation, GENERATION_COLUMNS),
            "### Correttezza per categoria", _category_table(generation, "correctness"),
            "### Validazione del giudice contro etichette umane", _judge_agreement(generation, labels_path),
        ]
    if latency:
        sections += ["## Latenza (dalle richieste di generazione)", _latency_table(latency)]
    (run_dir / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
