"""Orchestrazione di un run: indicizzazione, stadi di retrieval e generazione, report."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

from config import DATASETS_DIR, LLM_MODEL, MANIFEST_ENV_KEYS, RESULTS_DIR

from . import ollama, report
from .dataset import (
    Dataset,
    Question,
    build_zip,
    fetch_repo,
    list_repo_files,
    load_dataset,
    validate_dataset,
)
from .esperto_client import EspertoClient
from .metrics import gold_status, retrieval_metrics

logger = logging.getLogger("Benchmark")

_NAIVE = {"use_same_file": False, "use_symbol_graph": False, "use_symbol_lookup": False, "use_repo_map": False}

# nome -> (use_planner, opzioni di retrieval da disattivare). Ogni ablation differisce da `full` per un solo elemento.
CONFIGS: dict[str, tuple[bool, dict]] = {
    "full": (True, {}),
    "no_planner": (False, {}),
    "no_symbol_graph": (True, {"use_symbol_graph": False}),
    "no_same_file": (True, {"use_same_file": False}),
    "no_symbol_lookup": (True, {"use_symbol_lookup": False}),
    "no_repo_map": (True, {"use_repo_map": False}),  # la mappa guida il planner e apre il prompt
    "naive": (False, _NAIVE),
    # Stessa ricerca di `naive`, ma con K al massimo: è il budget di contesto a tagliare, come in
    # `full`. Dice se gli stadi aggiunti valgono più di "altro contesto preso in ordine denso".
    "naive_budget": (False, {**_NAIVE, "k": 50}),
}
CLOSED_BOOK = "closed_book"  # LLM senza contesto: misura quanto il modello conosce già il repo


def resolve_config(name: str) -> tuple[bool, dict]:
    """Risolve il nome di una configurazione.

    Il suffisso `@k=N` ne fissa il K (es. `full@k=20`).

    Args:
        name: Nome della configurazione, con il suffisso facoltativo.

    Returns:
        L'uso del planner e le opzioni di retrieval della configurazione.
    """
    base, _, k = name.partition("@k=")
    use_planner, options = CONFIGS[base]
    return use_planner, {**options, "k": int(k)} if k else options


class RunSpec(BaseModel):
    """Specifica di un run: dataset, stadi, configurazioni e campione di domande."""

    dataset: str
    stages: list[str] = ["retrieval", "generation"]
    retrieval_configs: list[str] = list(CONFIGS)
    # La generazione è lo stadio lento: di default solo sistema completo e le due baseline.
    generation_configs: list[str] = ["full", "naive", "naive_budget", CLOSED_BOOK]
    per_category: int | None = 4  # domande estratte per categoria; None = tutto il gold set
    seed: int = 0
    skip_index: bool = False
    planner: bool = True  # False: planner spento in tutte le configurazioni


def check_spec(spec: RunSpec) -> None:
    """Verifica che configurazioni e stadi di una specifica esistano.

    Args:
        spec: Specifica da verificare.

    Raises:
        ValueError: Se una configurazione o uno stadio sono sconosciuti.
    """
    for name in [*spec.retrieval_configs, *(n for n in spec.generation_configs if n != CLOSED_BOOK)]:
        try:
            resolve_config(name)
        except (KeyError, ValueError):
            raise ValueError(f"Configurazione sconosciuta: {name!r}") from None
    if set(spec.stages) - {"retrieval", "generation"}:
        raise ValueError(f"Stadi sconosciuti: {spec.stages}")


def sample_questions(questions: list[Question], per_category: int | None, seed: int) -> list[Question]:
    """Estrae un campione stratificato e riproducibile di domande.

    Args:
        questions: Domande del dataset.
        per_category: Domande da estrarre per categoria; None per tenerle tutte.
        seed: Seme dell'estrazione.

    Returns:
        Le domande estratte, nell'ordine del dataset.
    """
    if per_category is None:
        return questions
    chosen: set[str] = set()
    for category in sorted({question.category for question in questions}):
        ids = [question.id for question in questions if question.category == category]
        chosen.update(random.Random(f"{seed}:{category}").sample(ids, min(per_category, len(ids))))
    return [question for question in questions if question.id in chosen]


def _done(path: Path) -> set[tuple[str, str]]:
    """Legge le coppie già completate senza errori: una ripresa non le rifà.

    Args:
        path: File JSON Lines dello stadio.

    Returns:
        Le coppie (configurazione, ID della domanda) completate.
    """
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(row["config"], row["id"]) for row in rows if not row.get("error")}


def _compact(sources: list[dict]) -> list[dict]:
    """Riduce le fonti ai soli metadati, senza il testo dei chunk.

    Args:
        sources: Fonti restituite da esperto.

    Returns:
        Le fonti con i soli campi salvati nei risultati.
    """
    keys = (
        "filename", "symbol_name", "chunk_type", "start_line", "end_line", "chunk_id", "score", "origin",
        "graph_relation",
    )
    return [{key: source.get(key) for key in keys} for source in sources]


def _append(path: Path, row: dict) -> None:
    """Aggiunge una riga a un file JSON Lines.

    Args:
        path: File di destinazione.
        row: Oggetto da scrivere.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class _Status:
    """File `status.json` del run, riscritto a ogni aggiornamento."""

    def __init__(self, run_dir: Path, run_id: str):
        """Crea lo stato iniziale del run e lo scrive su file.

        Args:
            run_dir: Cartella del run.
            run_id: ID del run.
        """
        self._path = run_dir / "status.json"
        self._data = {"run_id": run_id, "status": "running", "stage": "setup", "done": 0, "total": 0, "error": None}
        self.update()

    def update(self, **fields) -> None:
        """Aggiorna lo stato e lo riscrive su file.

        Args:
            **fields: Campi da modificare.
        """
        self._data.update(fields)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def advance(self) -> None:
        """Segna come completata un'altra domanda."""
        self.update(done=self._data["done"] + 1)


def _manifest(
    run_id: str, spec: RunSpec, dataset: Dataset, commit: str, job: dict | None, health: dict, sampled: int
) -> dict:
    """Compone il manifest del run: tutto ciò che serve a riprodurlo.

    Args:
        run_id: ID del run.
        spec: Specifica del run.
        dataset: Dataset usato.
        commit: Commit del checkout del repository.
        job: Job di indicizzazione; None se l'indicizzazione è stata saltata.
        health: Stato di esperto all'avvio.
        sampled: Numero di domande del campione.

    Returns:
        Il manifest con specifica, dataset, parametri RAG, indicizzazione e versione di Python.
    """
    questions_bytes = (DATASETS_DIR / dataset.name / "questions.yaml").read_bytes()
    categories: dict[str, int] = {}
    for question in dataset.questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    return {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": spec.model_dump(),
        "dataset": {
            "name": dataset.name,
            "repo_url": dataset.repo_url,
            "ref": dataset.ref,
            "commit": commit,
            "questions_sha256": hashlib.sha256(questions_bytes).hexdigest(),
            "questions_by_category": categories,
            "sampled_questions": sampled,
        },
        "rag_params": {key: os.getenv(key) for key in MANIFEST_ENV_KEYS},
        "indexing_job": job,
        "esperto_health": health,
        "python": platform.python_version(),
    }


def _retrieval_stage(
    client: EspertoClient, spec: RunSpec, questions: list[Question], run_dir: Path, status: _Status
) -> None:
    """Esegue lo stadio di retrieval e ne scrive le righe in `retrieval.jsonl`.

    Args:
        client: Client di esperto.
        spec: Specifica del run.
        questions: Domande del campione.
        run_dir: Cartella del run.
        status: Stato del run, aggiornato a ogni domanda.
    """
    status.update(stage="retrieval", done=0, total=len(spec.retrieval_configs) * len(questions))
    done = _done(run_dir / "retrieval.jsonl")
    # Il piano del planner è calcolato una volta per domanda e riusato dalle ablation che lo
    # condividono: stessa scomposizione per tutte, e una sola chiamata al planner.
    plans: dict[str, list[str]] = {}
    for name in spec.retrieval_configs:
        use_planner, options = resolve_config(name)
        use_planner = use_planner and spec.planner
        shares_plan = use_planner and options.get("use_repo_map", True)
        for question in questions:
            status.advance()
            if (name, question.id) in done:
                continue
            row = {"config": name, "id": question.id, "category": question.category}
            try:
                plan = plans.get(question.id) if shares_plan else None
                result = client.retrieve(question.question, use_planner, options, plan)
                if shares_plan:
                    plans.setdefault(question.id, result["subquestions"])
                sources, dropped = result["sources"], result.get("dropped", [])
                row.update(
                    subquestions=result["subquestions"], sources=_compact(sources), dropped=dropped,
                    seconds=result["seconds"],
                )
                if question.relevant:
                    relevant = [item.model_dump() for item in question.relevant]
                    row["metrics"] = {**retrieval_metrics(sources, relevant), "seconds": result["seconds"]}
                    # Dove finisce ogni simbolo atteso: è la base della diagnosi del report.
                    row["gold"] = gold_status(sources, dropped, relevant)
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            _append(run_dir / "retrieval.jsonl", row)


def _write_judging_file(run_dir: Path, run_id: str, questions: list[Question]) -> None:
    """Scrive `to_judge.jsonl`: le risposte da giudicare, senza configurazione e in ordine mescolato.

    Args:
        run_dir: Cartella del run.
        run_id: ID del run.
        questions: Domande del campione.
    """
    by_id = {question.id: question for question in questions}
    lines = (run_dir / "generation.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    items = {
        (row["config"], row["id"]): {
            "key": report.judging_key(run_id, row["config"], row["id"]),
            "question": by_id[row["id"]].question,
            "key_facts": by_id[row["id"]].key_facts,
            "answer": row["answer"],
        }
        for row in rows
        if not row.get("error") and row["id"] in by_id
    }
    ordered = sorted(items.values(), key=lambda item: item["key"])
    (run_dir / "to_judge.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8"
    )


def _generate(client: EspertoClient, spec: RunSpec, dataset: Dataset, question: Question, name: str) -> dict:
    """Genera la risposta a una domanda con una configurazione.

    Args:
        client: Client di esperto.
        spec: Specifica del run.
        dataset: Dataset usato.
        question: Domanda da porre.
        name: Nome della configurazione; `closed_book` interroga il modello senza contesto.

    Returns:
        La risposta con le sue fonti e, per le configurazioni di esperto, tempi ed eventuale errore.
    """
    if name == CLOSED_BOOK:
        prompt = f"Rispondi in italiano a questa domanda sul progetto Python '{dataset.name}':\n{question.question}"
        return {"answer": ollama.chat(prompt, model=LLM_MODEL), "sources": []}
    use_planner, options = resolve_config(name)
    return client.ask(question.question, use_planner and spec.planner, options)


def _generation_stage(
    client: EspertoClient, spec: RunSpec, dataset: Dataset, questions: list[Question], run_dir: Path, status: _Status
) -> None:
    """Esegue lo stadio di generazione e ne scrive le righe in `generation.jsonl`.

    Args:
        client: Client di esperto.
        spec: Specifica del run.
        dataset: Dataset usato.
        questions: Domande del campione.
        run_dir: Cartella del run.
        status: Stato del run, aggiornato a ogni domanda.

    Raises:
        RuntimeError: Se esperto non risponde alla domanda di riscaldamento.
    """
    status.update(stage="generation", done=0, total=len(spec.generation_configs) * len(questions))
    done = _done(run_dir / "generation.jsonl")
    if set(spec.generation_configs) - {CLOSED_BOOK}:
        # Carica BGE-M3 e il modello prima di misurare i tempi; se fallisce, il run si ferma subito.
        warmup = client.ask(questions[0].question, spec.planner, {})
        if warmup["error"]:
            raise RuntimeError(f"Esperto non risponde: {warmup['error']}")
    for name in spec.generation_configs:
        for question in questions:
            status.advance()
            if (name, question.id) in done:
                continue
            row = {"config": name, "id": question.id, "category": question.category}
            try:
                result = _generate(client, spec, dataset, question, name)
                answer, sources = result["answer"], result["sources"]
                # La mappa del repository è nel prompt: fa parte di ciò che il modello ha visto.
                context = "\n\n---\n\n".join(
                    filter(None, [result.get("repo_map"), *(source["content"] for source in sources)])
                )
                row.update(answer=answer, sources=_compact(sources), error=result.get("error"))
                if row["error"]:  # stream interrotto: una risposta parziale non va giudicata
                    raise ValueError(row["error"])
                if context:  # per capire, davanti a un fatto sbagliato, se l'informazione c'era
                    contexts = run_dir / "contexts"
                    contexts.mkdir(exist_ok=True)
                    context_file = contexts / f"{report.judging_key(run_dir.name, name, question.id)}.txt"
                    context_file.write_text(context, encoding="utf-8")
                # I tempi vengono dalla stessa richiesta: nessuno stadio di latenza separato.
                timings = {k: result[k] for k in ("retrieval_s", "ttft_s", "total_s") if result.get(k) is not None}
                row["metrics"] = {**timings, "answer_error": 0.0, "answer_chars": float(len(answer))}
                if question.relevant and name != CLOSED_BOOK:
                    # Serve a separare gli errori del retrieval da quelli del modello.
                    relevant = [item.model_dump() for item in question.relevant]
                    recall = retrieval_metrics(sources, relevant)["symbol_context_recall"]
                    row["metrics"]["symbol_context_recall"] = recall
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["metrics"] = {"answer_error": 1.0}
            _append(run_dir / "generation.jsonl", row)


def execute_run(run_id: str, spec: RunSpec, client: EspertoClient | None = None) -> None:
    """Esegue il run e ne scrive tutti i file in RESULTS_DIR/<run_id>.

    Gli errori non si propagano: finiscono in `status.json`.

    Args:
        run_id: ID del run.
        spec: Specifica del run.
        client: Client di esperto; se manca ne viene creato uno.
    """
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = _Status(run_dir, run_id)
    try:
        client = client or EspertoClient()
        dataset = load_dataset(spec.dataset)
        questions = sample_questions(dataset.questions, spec.per_category, spec.seed)
        repo_dir, commit = fetch_repo(dataset)
        errors = validate_dataset(dataset, repo_dir)
        if errors:
            raise ValueError("Gold set non valido: " + "; ".join(errors[:5]))
        repo_files = list_repo_files(repo_dir)

        job = None
        zip_name = f"{dataset.name}.zip"
        if not spec.skip_index:
            status.update(stage="indexing")
            job = client.index_zip(build_zip(repo_dir, repo_files), zip_name)
        elif client.indexed_filename() != zip_name:
            raise ValueError(f"L'indice attivo non è quello di {zip_name}: rilanciare senza skip_index")
        health = client.health()
        if health.get("status") != "ok":
            raise RuntimeError(f"Esperto non pronto: {health}")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():  # in una ripresa resta il manifest del run originale
            manifest = _manifest(run_id, spec, dataset, commit, job, health, len(questions))
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if "retrieval" in spec.stages:
            _retrieval_stage(client, spec, questions, run_dir, status)
        if "generation" in spec.stages:
            _generation_stage(client, spec, dataset, questions, run_dir, status)
            _write_judging_file(run_dir, run_id, questions)

        status.update(stage="report")
        report.write_report(run_dir)
        status.update(status="done", stage="done")
    except Exception as error:  # il thread non deve morire in silenzio: l'errore va in status.json
        logger.exception("Run %s fallito", run_id)
        status.update(status="error", error=f"{type(error).__name__}: {error}")
