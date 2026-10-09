"""Report di un run: diagnosi, tabelle, summary e confronto tra run.

Il report risponde a tre domande: dove si perdono i simboli attesi (ricerca o budget), quanto
conta ogni stadio (ablation), e se un fatto sbagliato dipende dal retrieval o dal modello.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from .metrics import LOST, ORIGINS, bootstrap_ci, mean, paired_bootstrap_diff, percentile, share

# (metrica, etichetta, formato): senza formato la cella mostra media e intervallo di confidenza.
RETRIEVAL_COLUMNS = [
    ("symbol_mrr", "MRR", None),
    ("symbol_context_recall", "Ctx recall", None),
    ("hop2_symbol_recall", "Hop-2 recall", None),
    ("n_sources", "Fonti", "{:.1f}"),
    ("context_chars", "Caratteri", "{:.0f}"),
    ("seconds", "Secondi", "{:.2f}"),
]
GENERATION_COLUMNS = [
    ("correctness", "Correttezza", None),
    ("hallucination", "Risposte inventate (senza risposta)", "{:.2f}"),
    ("false_refusal", "Rifiuti errati", "{:.2f}"),
    ("answer_error", "Errori", "{:.2f}"),
    ("judged", "Giudicate", "{:.2f}"),
    ("answer_chars", "Caratteri risposta", "{:.0f}"),
]
LATENCY_COLUMNS = [("retrieval_s", "Retrieval (s)"), ("ttft_s", "TTFT (s)"), ("total_s", "Totale (s)")]
DELTA_METRICS = {"symbol_mrr": "MRR", "symbol_context_recall": "ctx recall", "hop2_symbol_recall": "hop-2"}
COMPARE_METRICS = [
    ("retrieval", "symbol_mrr"), ("retrieval", "symbol_context_recall"), ("retrieval", "hop2_symbol_recall"),
    ("generation", "correctness"), ("generation", "total_s"),
]
ORIGIN_LABELS = {"search": "ricerca", "symbol": "simboli nominati", "same_file": "stesso file", "graph": "grafo"}
# Configurazione di ablation -> stadio che spegne: la diagnosi dice quali stadi contano.
STAGE_OF = {
    "no_planner": "planner", "no_symbol_graph": "grafo", "no_same_file": "stesso file",
    "no_symbol_lookup": "simboli nominati", "dense_only": "ricerca sparsa", "sparse_only": "ricerca densa",
    "no_repo_map": "mappa nel planner",
}
MAX_ROWS = 20


# --- Caricamento ---

def _load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load(path: Path) -> list[dict]:
    # Dopo una ripresa una domanda puo' comparire due volte: vale l'ultimo tentativo.
    return list({(row["config"], row["id"]): row for row in _load_lines(path)}.values())


def _apply_judgments(run_dir: Path, run_id: str, generation: list[dict]) -> None:
    """Unisce i verdetti di `judgments.jsonl` alle risposte e ne ricava le metriche."""
    judgments = {row["key"]: row for row in _load_lines(run_dir / "judgments.jsonl")}
    facts = {row["key"]: row["key_facts"] for row in _load_lines(run_dir / "to_judge.jsonl")}
    for row in generation:
        if row.get("error"):
            continue
        key = hashlib.sha1(f"{run_id}:{row['config']}:{row['id']}".encode("utf-8")).hexdigest()[:12]
        judgment = judgments.get(key)
        # Un verdetto con un numero di fatti sbagliato viene scartato: la risposta resta non giudicata.
        valid = judgment is not None and key in facts and len(judgment["fact_verdicts"]) == len(facts[key])
        row["metrics"]["judged"] = float(valid)
        if not valid:
            continue
        # Risposta inventata: risposta data a una domanda che non ne ha; rifiuto errato: il contrario.
        if row["category"] == "unanswerable":
            row["metrics"]["hallucination"] = float(not judgment["abstains"])
        else:
            row["fact_verdicts"], row["key_facts"] = judgment["fact_verdicts"], facts[key]
            row["metrics"]["correctness"] = mean(judgment["fact_verdicts"])
            row["metrics"]["false_refusal"] = float(judgment["abstains"])


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Manifest, righe di retrieval e righe di generazione (con i verdetti applicati) di un run."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    generation = _load(run_dir / "generation.jsonl")
    _apply_judgments(run_dir, manifest["run_id"], generation)
    return manifest, _load(run_dir / "retrieval.jsonl"), generation


# --- Aggregazione ---

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
        "n": len(values), "mean": mean(values), "ci_lo": low, "ci_hi": high,
        "p50": percentile(values, 50), "p95": percentile(values, 95),
    }


def _reference(rows: list[dict]) -> str:
    """Configurazione a cui si riferiscono diagnosi e differenze: `full`, o la prima presente."""
    configs = _configs(rows)
    return "full" if "full" in configs or not configs else configs[0]


def _delta(rows: list[dict], config: str, other: str, metric: str) -> Optional[tuple[float, float, float]]:
    """Differenza appaiata `config` - `other` sulle domande comuni, con CI 95%."""
    a, b = _by_id(rows, config, metric), _by_id(rows, other, metric)
    common = sorted(set(a) & set(b))
    return paired_bootstrap_diff([a[i] for i in common], [b[i] for i in common]) if len(common) > 1 else None


def _significant(delta: Optional[tuple[float, float, float]]) -> bool:
    return delta is not None and (delta[1] > 0 or delta[2] < 0)


# --- Celle e tabelle ---

def _cell(values: list[float], number_format: Optional[str] = None) -> str:
    stat = _stat(values)
    if stat["n"] == 0:
        return "n/d"
    if number_format:
        return number_format.format(stat["mean"])
    return f"{stat['mean']:.2f} [{stat['ci_lo']:.2f}, {stat['ci_hi']:.2f}]"


def _delta_cell(delta: Optional[tuple[float, float, float]]) -> str:
    if delta is None:
        return "n/d"
    return f"{delta[0]:+.2f} [{delta[1]:+.2f}, {delta[2]:+.2f}]{'*' if _significant(delta) else ''}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _metric_table(rows: list[dict], columns: list[tuple]) -> str:
    return _table(
        ["Config", "N"] + [label for _, label, _ in columns],
        [
            [c, str(sum(row["config"] == c for row in rows))] + [_cell(_values(rows, c, m), f) for m, _, f in columns]
            for c in _configs(rows)
        ],
    )


def _category_table(rows: list[dict], metric: str) -> str:
    categories = sorted({row["category"] for row in rows if row.get("metrics", {}).get(metric) is not None})
    return _table(
        ["Config"] + categories,
        [[c] + [_cell(_values(rows, c, metric, category)) for category in categories] for c in _configs(rows)],
    )


def _delta_table(rows: list[dict], reference: str) -> str:
    return _table(
        [f"Config - {reference}"] + list(DELTA_METRICS.values()),
        [
            [config] + [_delta_cell(_delta(rows, config, reference, metric)) for metric in DELTA_METRICS]
            for config in _configs(rows)
            if config != reference
        ],
    )


def _latency_table(rows: list[dict]) -> str:
    def cell(config: str, metric: str) -> str:
        stat = _stat(_values(rows, config, metric))
        return "n/d" if stat["n"] == 0 else f"{stat['p50']:.2f} / {stat['p95']:.2f}"

    return _table(
        ["Config"] + [f"{label} p50 / p95" for _, label in LATENCY_COLUMNS],
        [[c] + [cell(c, m) for m, _ in LATENCY_COLUMNS] for c in _configs(rows)],
    )


# --- Dove si perdono i simboli attesi ---

def _gold(rows: list[dict], config: str, hop: Optional[int] = None) -> list[str]:
    """Stato di ogni simbolo atteso di una configurazione (vedi `metrics.gold_status`)."""
    return [
        item["status"]
        for row in rows
        if row["config"] == config
        for item in row.get("gold", [])
        if hop is None or item["hop"] == hop
    ]


def _funnel_table(rows: list[dict]) -> str:
    table = []
    for config in _configs(rows):
        statuses = _gold(rows, config)
        if statuses:
            cells = [share(statuses, ORIGINS), *(share(statuses, [o]) for o in ORIGINS), *(share(statuses, [s]) for s in LOST)]
            table.append([config, str(len(statuses))] + [f"{value:.0%}" for value in cells])
    headers = ["Config", "Simboli attesi", "Nel contesto"] + [f"di cui da {ORIGIN_LABELS[o]}" for o in ORIGINS]
    return _table(headers + ["Esclusi dal budget", "Mai trovati"], table) if table else ""


def _symbol_label(item: dict) -> str:
    filename, _, symbol = item["symbol"].partition("::")
    return f"`{symbol}` ({Path(filename).name}{', hop 2' if item['hop'] == 2 else ''})"


def _missing_table(rows: list[dict], reference: str) -> str:
    """Le domande della configurazione di riferimento con simboli attesi fuori dal contesto."""
    lost = [r for r in rows if r["config"] == reference and any(g["status"] in LOST for g in r.get("gold", []))]
    lost.sort(key=lambda row: (row["metrics"]["symbol_context_recall"], row["id"]))

    def names(row: dict, status: str) -> str:
        return ", ".join(_symbol_label(g) for g in row["gold"] if g["status"] == status) or "-"

    table = [
        [row["id"], row["category"], f"{row['metrics']['symbol_context_recall']:.2f}", names(row, "budget"), names(row, "missing")]
        for row in lost[:MAX_ROWS]
    ]
    more = f"\n\n... e altre {len(lost) - MAX_ROWS} domande (vedi `retrieval.jsonl`)." if len(lost) > MAX_ROWS else ""
    return _table(["Domanda", "Categoria", "Ctx recall", "Esclusi dal budget", "Mai trovati"], table) + more if table else ""


# --- Retrieval o modello? ---

def _with_context(generation: list[dict]) -> list[dict]:
    """Risposte date con un contesto (la baseline senza contesto non dice nulla sul retrieval)."""
    return [row for row in generation if row.get("sources")]


def _attribution(generation: list[dict], config: str) -> tuple[list[float], list[float]]:
    """Correttezza delle domande con contesto completo e incompleto (tutti i simboli attesi o no)."""
    rows = [
        row["metrics"]
        for row in generation
        if row["config"] == config and row.get("metrics", {}).get("correctness") is not None
        and row["metrics"].get("symbol_context_recall") is not None
    ]
    complete = [m["correctness"] for m in rows if m["symbol_context_recall"] == 1.0]
    return complete, [m["correctness"] for m in rows if m["symbol_context_recall"] < 1.0]


def _attribution_table(generation: list[dict]) -> str:
    table = []
    for config in _configs(_with_context(generation)):
        complete, partial = _attribution(generation, config)
        if complete or partial:
            table.append([config, str(len(complete)), _cell(complete, "{:.2f}"), str(len(partial)), _cell(partial, "{:.2f}")])
    headers = ["Config", "Contesto completo: domande", "correttezza", "Contesto incompleto: domande", "correttezza"]
    return _table(headers, table) if table else ""


def _wrong_facts(generation: list[dict]) -> list[dict]:
    """Fatti chiave sbagliati e configurazioni che li sbagliano (`°` = contesto incompleto)."""
    wrong: dict[tuple[str, int], dict] = {}
    for row in _with_context(generation):
        recall = row["metrics"].get("symbol_context_recall")
        incomplete = recall is not None and recall < 1.0
        for index, correct in enumerate(row.get("fact_verdicts", [])):
            if not correct:
                entry = wrong.setdefault((row["id"], index), {"id": row["id"], "fact": row["key_facts"][index], "configs": []})
                entry["configs"].append(row["config"] + ("°" if incomplete else ""))
    return sorted(wrong.values(), key=lambda entry: (-len(entry["configs"]), entry["id"]))


def _wrong_facts_table(generation: list[dict]) -> str:
    wrong = _wrong_facts(generation)
    table = [[entry["id"], entry["fact"], ", ".join(entry["configs"])] for entry in wrong[:MAX_ROWS]]
    more = f"\n\n... e altri {len(wrong) - MAX_ROWS} fatti (vedi `generation.jsonl`)." if len(wrong) > MAX_ROWS else ""
    return _table(["Domanda", "Fatto sbagliato", "Sbagliato in"], table) + more if table else ""


# --- Diagnosi ---

def _diagnosis(retrieval: list[dict], generation: list[dict]) -> list[str]:
    """Poche frasi ricavate dai dati: dove intervenire prima di leggere le tabelle."""
    points = []
    reference = _reference(retrieval)
    statuses = _gold(retrieval, reference)
    if statuses:
        cut, missing = share(statuses, ["budget"]), share(statuses, ["missing"])
        if max(cut, missing) < 0.05:
            hint = "Il retrieval porta nel contesto quasi tutto ciò che serve."
        elif cut > missing:
            hint = "Il limite principale è il budget: i chunk giusti vengono trovati ma non entrano (ordinamento, dimensione dei chunk, budget)."
        else:
            hint = "Il limite principale è la ricerca: i chunk giusti non diventano candidati (ricerca, simboli nominati, grafo)."
        points.append(
            f"**Simboli attesi** (`{reference}`): {share(statuses, ORIGINS):.0%} nel contesto, "
            f"{cut:.0%} trovati ma esclusi dal budget, {missing:.0%} mai trovati. {hint}"
        )
        hop2 = _gold(retrieval, reference, hop=2)
        if hop2:
            points.append(
                f"**Simboli a un passo di distanza** (hop 2, {len(hop2)}): {share(hop2, ORIGINS):.0%} nel contesto "
                f"({share(hop2, ['graph']):.0%} grazie al grafo), {share(hop2, ['budget']):.0%} esclusi dal budget, "
                f"{share(hop2, ['missing']):.0%} mai trovati."
            )
    useful, neutral = [], []
    for config in _configs(retrieval):
        if config not in STAGE_OF or config == reference:
            continue
        deltas = {label: _delta(retrieval, config, reference, metric) for metric, label in DELTA_METRICS.items()}
        effects = [f"{label} {delta[0]:+.2f}" for label, delta in deltas.items() if _significant(delta)]
        if effects:
            useful.append(f"{STAGE_OF[config]} ({', '.join(effects)} senza)")
        else:
            neutral.append(STAGE_OF[config])
    if useful:
        points.append("**Stadi con effetto misurabile**: " + "; ".join(useful) + ".")
    if neutral:
        points.append(
            "**Stadi senza effetto misurabile**: " + ", ".join(neutral)
            + ". Sono i candidati da rivedere o togliere; differenze piccole non si distinguono con questo numero di domande."
        )
    for baseline in [c for c in _configs(retrieval) if c.startswith("naive") and c != reference]:
        cells = [f"{label} {_delta_cell(_delta(retrieval, reference, baseline, metric))}" for metric, label in DELTA_METRICS.items()]
        points.append(f"**`{reference}` rispetto a `{baseline}`**: " + ", ".join(cells) + ".")

    answered = _with_context(generation)
    reference = _reference(answered)
    complete, partial = _attribution(generation, reference)
    if complete or partial:
        wrong = [
            (row["metrics"]["symbol_context_recall"] or 0.0) < 1.0
            for row in answered
            if row["config"] == reference and row["metrics"].get("symbol_context_recall") is not None
            for correct in row.get("fact_verdicts", [])
            if not correct
        ]
        text = (
            f"**Retrieval o modello?** (`{reference}`) Con tutti i simboli attesi nel contesto la correttezza è "
            f"{_cell(complete, '{:.2f}')} ({len(complete)} domande); con contesto incompleto {_cell(partial, '{:.2f}')} "
            f"({len(partial)}). Fatti sbagliati in queste domande: {len(wrong)}, di cui {sum(wrong)} con contesto incompleto "
            f"(da cercare nel retrieval) e {len(wrong) - sum(wrong)} con contesto completo (modello o prompt)."
        )
        points.append(text)
    wrong_facts, configs = _wrong_facts(generation), _configs(answered)
    everywhere = sum(len(entry["configs"]) == len(configs) for entry in wrong_facts)
    if wrong_facts and len(configs) > 1:
        points.append(
            f"**Fatti sbagliati da tutte le configurazioni**: {everywhere} su {len(wrong_facts)}. "
            "Non dipendono dalla configurazione: vedi l'elenco in fondo."
        )
    return points


# --- Summary ---

def _summary(stage: str, rows: list[dict]) -> list[dict]:
    entries = []
    metrics = sorted({name for row in rows for name in row.get("metrics", {})})
    for category in ["all", *sorted({row["category"] for row in rows})]:
        for config in _configs(rows):
            for metric in metrics:
                stat = _stat(_values(rows, config, metric, None if category == "all" else category))
                if stat["n"]:
                    entries.append({"stage": stage, "config": config, "category": category, "metric": metric, **stat})
    return entries


def _write_summary(run_dir: Path, entries: list[dict]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]) if entries else ["stage"])
        writer.writeheader()
        writer.writerows(entries)


# --- Report e confronto ---

def _header(manifest: dict, judged: int) -> str:
    dataset, job, params = manifest["dataset"], manifest.get("indexing_job") or {}, manifest["rag_params"]
    spec = manifest["spec"]
    return "\n".join([
        f"- Dataset: `{dataset['name']}` ({dataset['repo_url']} @ `{dataset['ref']}`, commit `{dataset['commit'][:10]}`)",
        f"- Campione: {dataset.get('sampled_questions', 'n/d')} domande"
        f" (per categoria: {spec.get('per_category') or 'tutte'}, seed {spec.get('seed', 0)});"
        f" planner {'acceso' if spec.get('planner', True) else 'spento'}",
        f"- Parametri: modello `{params.get('LLM_MODEL')}`, K={params.get('K')},"
        f" soglia={params.get('SEMANTIC_RELEVANCE_RATIO')}, finestra={params.get('LLM_CONTEXT_TOKENS')} token",
        f"- Indicizzazione: {job.get('indexing_seconds', 'n/d')} s, {manifest['esperto_health'].get('points_count', 'n/d')} punti",
        f"- Risposte giudicate: {judged} (`to_judge.jsonl` -> `POST /runs/{{id}}/judgments`)",
        "- Celle: media [IC 95% bootstrap sulle domande]; `*` = differenza appaiata con IC che esclude 0.",
    ])


def write_report(run_dir: Path) -> None:
    manifest, retrieval_rows, generation = load_run(run_dir)
    retrieval = [row for row in retrieval_rows if "metrics" in row]
    judged = int(sum(row.get("metrics", {}).get("judged", 0) for row in generation))
    reference = _reference(retrieval)
    _write_summary(run_dir, _summary("retrieval", retrieval) + _summary("generation", generation))

    def section(title: str, body: str) -> list[str]:
        return [title, body] if body else []

    sections = [f"# Benchmark {manifest['run_id']}", _header(manifest, judged)]
    diagnosis = _diagnosis(retrieval, generation)
    if diagnosis:
        sections += ["## Diagnosi", "\n".join(f"- {point}" for point in diagnosis)]
    if retrieval:
        failed = sum(bool(row.get("error")) for row in retrieval_rows)
        sections += ["## Retrieval"]
        sections += [f"**Attenzione**: {failed} richieste fallite su {len(retrieval_rows)}, escluse dalle medie."] if failed else []
        sections += [_metric_table(retrieval, RETRIEVAL_COLUMNS)]
        sections += section(
            "### Dove finiscono i simboli attesi",
            _funnel_table(retrieval) and "Quote sul totale dei simboli attesi (il ctx recall è invece la media per domanda). "
            "\"Esclusi dal budget\": trovati dalla ricerca ma rimasti fuori dal contesto.\n\n" + _funnel_table(retrieval),
        )
        if len(_configs(retrieval)) > 1:
            sections += section(f"### Effetto di ogni configurazione (differenza rispetto a `{reference}`)", _delta_table(retrieval, reference))
        sections += section("### Ctx recall per categoria", _category_table(retrieval, "symbol_context_recall"))
        sections += section(f"### Domande con simboli mancanti (`{reference}`)", _missing_table(retrieval, reference))
    if generation:
        sections += ["## Generazione", _metric_table(generation, GENERATION_COLUMNS)]
        if judged:
            sections += section("### Correttezza per categoria", _category_table(generation, "correctness"))
            sections += section("### Retrieval o modello?", _attribution_table(generation))
            sections += section("### Fatti sbagliati (`°` = contesto incompleto per quella domanda)", _wrong_facts_table(generation))
        if any("total_s" in row.get("metrics", {}) for row in generation):
            sections += ["### Latenza", _latency_table([r for r in generation if "total_s" in r.get("metrics", {})])]
    (run_dir / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def compare_runs(run_dir: Path, other_dir: Path) -> str:
    """Differenze appaiate tra due run, sulle configurazioni e sulle domande che hanno in comune."""
    stages = dict(zip(("retrieval", "generation"), load_run(run_dir)[1:]))
    other_stages = dict(zip(("retrieval", "generation"), load_run(other_dir)[1:]))
    table = []
    for stage, metric in COMPARE_METRICS:
        for config in _configs(stages[stage]):
            a, b = _by_id(stages[stage], config, metric), _by_id(other_stages[stage], config, metric)
            common = sorted(set(a) & set(b))
            if len(common) > 1:
                delta = paired_bootstrap_diff([a[i] for i in common], [b[i] for i in common])
                table.append([
                    metric, config, f"{mean([b[i] for i in common]):.2f}", f"{mean([a[i] for i in common]):.2f}",
                    _delta_cell(delta), str(len(common)),
                ])
    title = f"# Confronto: {run_dir.name} rispetto a {other_dir.name}"
    if not table:
        return f"{title}\n\nNessuna configurazione e domanda in comune tra i due run.\n"
    note = "Differenza = questo run - l'altro, sulle stesse domande; `*` = IC 95% che esclude 0."
    return f"{title}\n\n{note}\n\n" + _table(["Metrica", "Config", other_dir.name, run_dir.name, "Differenza", "Domande"], table) + "\n"
