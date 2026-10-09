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
from typing import List, Optional

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

_NAIVE = {
    "use_sparse": False, "use_same_file": False,
    "use_symbol_graph": False, "use_symbol_lookup": False, "use_repo_map": False,
}

# nome -> (use_planner, opzioni di retrieval da disattivare). Ogni ablation differisce da `full` per un solo elemento.
CONFIGS: dict[str, tuple[bool, dict]] = {
    "full": (True, {}),
    "no_planner": (False, {}),
    "no_symbol_graph": (True, {"use_symbol_graph": False}),
    "no_same_file": (True, {"use_same_file": False}),
    "no_symbol_lookup": (True, {"use_symbol_lookup": False}),
    "dense_only": (True, {"use_sparse": False}),
    "sparse_only": (True, {"use_dense": False}),
    "no_repo_map": (True, {"use_repo_map": False}),  # la mappa guida il planner e apre il prompt
    "naive": (False, _NAIVE),
    # Stessa ricerca di `naive`, ma con K al massimo: e' il budget di contesto a tagliare, come in
    # `full`. Dice se gli stadi aggiunti valgono piu' di "altro contesto preso in ordine denso".
    "naive_budget": (False, {**_NAIVE, "k": 50}),
}
CLOSED_BOOK = "closed_book"  # LLM senza contesto: misura quanto il modello conosce gia' il repo


def resolve_config(name: str) -> tuple[bool, dict]:
    """(use_planner, opzioni) di una configurazione; `nome@k=N` ne fissa il K (es. `full@k=20`)."""
    base, _, k = name.partition("@k=")
    use_planner, options = CONFIGS[base]
    return use_planner, {**options, "k": int(k)} if k else options


class RunSpec(BaseModel):
    dataset: str
    stages: List[str] = ["retrieval", "generation"]
    retrieval_configs: List[str] = list(CONFIGS)
    # La generazione e' lo stadio lento: di default solo sistema completo e le due baseline.
    generation_configs: List[str] = ["full", "naive", "naive_budget", CLOSED_BOOK]
    per_category: Optional[int] = 4  # domande estratte per categoria; None = tutto il gold set
    seed: int = 0
    skip_index: bool = False
    planner: bool = True  # False: planner spento in tutte le configurazioni


def check_spec(spec: RunSpec) -> None:
    for name in [*spec.retrieval_configs, *(n for n in spec.generation_configs if n != CLOSED_BOOK)]:
        try:
            resolve_config(name)
        except (KeyError, ValueError):
            raise ValueError(f"Configurazione sconosciuta: {name!r}") from None
    if set(spec.stages) - {"retrieval", "generation"}:
        raise ValueError(f"Stadi sconosciuti: {spec.stages}")


def sample_questions(questions: list[Question], per_category: Optional[int], seed: int) -> list[Question]:
    """Campione stratificato e riproducibile: `per_category` domande per categoria, nell'ordine del dataset."""
    if per_category is None:
        return questions
    chosen: set[str] = set()
    for category in sorted({question.category for question in questions}):
        ids = [question.id for question in questions if question.category == category]
        chosen.update(random.Random(f"{seed}:{category}").sample(ids, min(per_category, len(ids))))
    return [question for question in questions if question.id in chosen]


def _done(path: Path) -> set[tuple[str, str]]:
    """Coppie (config, domanda) gia' completate senza errori: una ripresa non le rifa'."""
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(row["config"], row["id"]) for row in rows if not row.get("error")}


def _compact(sources: list[dict]) -> list[dict]:
    keys = ("filename", "symbol_name", "chunk_type", "start_line", "end_line", "chunk_id", "score", "origin", "graph_relation")
    return [{key: source.get(key) for key in keys} for source in sources]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class _Status:
    """status.json del run, riscritto a ogni aggiornamento."""

    def __init__(self, run_dir: Path, run_id: str):
        self._path = run_dir / "status.json"
        self._data = {"run_id": run_id, "status": "running", "stage": "setup", "done": 0, "total": 0, "error": None}
        self.update()

    def update(self, **fields) -> None:
        self._data.update(fields)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def advance(self) -> None:
        self.update(done=self._data["done"] + 1)


def _manifest(
    run_id: str, spec: RunSpec, dataset: Dataset, commit: str, job: Optional[dict], health: dict, sampled: int
) -> dict:
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


def _retrieval_stage(client: EspertoClient, spec: RunSpec, questions: list[Question], run_dir: Path, status: _Status) -> None:
    status.update(stage="retrieval", done=0, total=len(spec.retrieval_configs) * len(questions))
    done = _done(run_dir / "retrieval.jsonl")
    # Il piano del planner e' calcolato una volta per domanda e riusato dalle ablation che lo
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
                    # Dove finisce ogni simbolo atteso: e' la base della diagnosi del report.
                    row["gold"] = gold_status(sources, dropped, relevant)
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            _append(run_dir / "retrieval.jsonl", row)


def judging_key(run_id: str, config: str, question_id: str) -> str:
    """Chiave opaca di una risposta: chi giudica non vede da quale configurazione viene."""
    return hashlib.sha1(f"{run_id}:{config}:{question_id}".encode("utf-8")).hexdigest()[:12]


def _write_judging_file(run_dir: Path, run_id: str, questions: list[Question]) -> None:
    """`to_judge.jsonl`: le risposte da giudicare, senza configurazione e in ordine mescolato."""
    by_id = {question.id: question for question in questions}
    rows = [json.loads(line) for line in (run_dir / "generation.jsonl").read_text(encoding="utf-8").splitlines() if line]
    items = {
        (row["config"], row["id"]): {
            "key": judging_key(run_id, row["config"], row["id"]),
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
    """Risposta + contesto per una configurazione (closed_book: nessun contesto)."""
    if name == CLOSED_BOOK:
        prompt = f"Rispondi in italiano a questa domanda sul progetto Python '{dataset.name}':\n{question.question}"
        return {"answer": ollama.chat(prompt, model=LLM_MODEL), "sources": []}
    use_planner, options = resolve_config(name)
    return client.ask(question.question, use_planner and spec.planner, options)


def _generation_stage(
    client: EspertoClient, spec: RunSpec, dataset: Dataset, questions: list[Question], run_dir: Path, status: _Status
) -> None:
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
                # La mappa del repository e' nel prompt: fa parte di cio' che il modello ha visto.
                context = "\n\n---\n\n".join(
                    filter(None, [result.get("repo_map"), *(source["content"] for source in sources)])
                )
                row.update(answer=answer, sources=_compact(sources), error=result.get("error"))
                if row["error"]:  # stream interrotto: una risposta parziale non va giudicata
                    raise ValueError(row["error"])
                if context:  # per capire, davanti a un fatto sbagliato, se l'informazione c'era
                    contexts = run_dir / "contexts"
                    contexts.mkdir(exist_ok=True)
                    (contexts / f"{judging_key(run_dir.name, name, question.id)}.txt").write_text(context, encoding="utf-8")
                # I tempi vengono dalla stessa richiesta: nessuno stadio di latenza separato.
                timings = {k: result[k] for k in ("retrieval_s", "ttft_s", "total_s") if result.get(k) is not None}
                row["metrics"] = {**timings, "answer_error": 0.0, "answer_chars": float(len(answer))}
                if question.relevant and name != CLOSED_BOOK:
                    # Serve a separare gli errori del retrieval da quelli del modello.
                    relevant = [item.model_dump() for item in question.relevant]
                    row["metrics"]["symbol_context_recall"] = retrieval_metrics(sources, relevant)["symbol_context_recall"]
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["metrics"] = {"answer_error": 1.0}
            _append(run_dir / "generation.jsonl", row)


def execute_run(run_id: str, spec: RunSpec, client: Optional[EspertoClient] = None) -> None:
    """Esegue il run e scrive tutti gli artefatti in RESULTS_DIR/<run_id>. Errori in status.json."""
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
            raise ValueError(f"L'indice attivo non e' quello di {zip_name}: rilanciare senza skip_index")
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
