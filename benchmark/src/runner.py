"""Orchestrazione di un run: indicizzazione, stadi retrieval/generazione/latenza, report."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel

from config import DATASETS_DIR, JUDGE_MODEL, LLM_MODEL, MANIFEST_ENV_KEYS, RESULTS_DIR

from . import judge, ollama, report
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
from .metrics import citation_stats, retrieval_metrics

logger = logging.getLogger("Benchmark")

# nome -> (use_planner, opzioni di retrieval da disattivare). Ogni ablation differisce da `full` per un solo elemento.
CONFIGS: dict[str, tuple[bool, dict]] = {
    "full": (True, {}),
    "no_planner": (False, {}),
    "no_symbol_graph": (True, {"use_symbol_graph": False}),
    "no_same_file": (True, {"use_same_file": False}),
    "no_colbert": (True, {"use_colbert": False}),
    "dense_only": (True, {"use_sparse": False}),
    "sparse_only": (True, {"use_dense": False}),
    "no_repo_map": (True, {"use_repo_map": False}),  # la mappa guida il planner e apre il prompt
    "naive": (
        False,
        {"use_sparse": False, "use_colbert": False, "use_same_file": False, "use_symbol_graph": False, "use_repo_map": False},
    ),
}
CLOSED_BOOK = "closed_book"  # LLM senza contesto: misura quanto il modello conosce gia' il repo


class RunSpec(BaseModel):
    dataset: str
    stages: List[str] = ["retrieval", "generation", "latency"]
    retrieval_configs: List[str] = list(CONFIGS)
    generation_configs: List[str] = ["full", "no_planner", "no_symbol_graph", "naive", CLOSED_BOOK]
    limit: Optional[int] = None  # solo le prime N domande (smoke test)
    latency_questions: int = 20
    latency_repeats: int = 3
    latency_warmups: int = 2
    skip_index: bool = False


def check_spec(spec: RunSpec) -> None:
    valid = set(CONFIGS) | {CLOSED_BOOK}
    unknown = (set(spec.retrieval_configs) - set(CONFIGS)) | (set(spec.generation_configs) - valid)
    if unknown or set(spec.stages) - {"retrieval", "generation", "latency"}:
        raise ValueError(f"Configurazioni o stadi sconosciuti: {sorted(unknown) or spec.stages}")


def _compact(sources: list[dict]) -> list[dict]:
    keys = ("filename", "symbol_name", "chunk_type", "start_line", "end_line", "chunk_id", "score", "graph_relation")
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


def _manifest(run_id: str, spec: RunSpec, dataset: Dataset, commit: str, job: Optional[dict], health: dict) -> dict:
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
        },
        "judge_model": JUDGE_MODEL,
        "rag_params": {key: os.getenv(key) for key in MANIFEST_ENV_KEYS},
        "indexing_job": job,
        "esperto_health": health,
        "python": platform.python_version(),
    }


def _retrieval_stage(client: EspertoClient, spec: RunSpec, questions: list[Question], run_dir: Path, status: _Status) -> None:
    status.update(stage="retrieval", done=0, total=len(spec.retrieval_configs) * len(questions))
    for name in spec.retrieval_configs:
        use_planner, options = CONFIGS[name]
        for question in questions:
            row = {"config": name, "id": question.id, "category": question.category}
            try:
                result = client.retrieve(question.question, use_planner, options)
                row.update(subquestions=result["subquestions"], sources=_compact(result["sources"]), seconds=result["seconds"])
                if question.answerable:
                    row["metrics"] = retrieval_metrics(result["sources"], [r.model_dump() for r in question.relevant])
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            _append(run_dir / "retrieval.jsonl", row)
            status.advance()


def _judge_answer(question: Question, answer: str, context: str, has_context: bool) -> dict:
    """Verdetti del giudice per una risposta; None = il giudice ha fallito su quel verdetto."""
    metrics: dict = {}
    verdicts = judge.judge_facts(question.question, answer, question.key_facts) if question.answerable else None
    if verdicts is not None:
        metrics["correctness"] = sum(verdicts) / len(verdicts)
    abstains = judge.judge_abstains(question.question, answer)
    if abstains is not None:
        # hallucination: risposta data a una domanda che non ha risposta; false_refusal: il contrario.
        if question.answerable:
            metrics["false_refusal"] = float(abstains)
        else:
            metrics["hallucination"] = float(not abstains)
    if has_context:
        faithful = judge.judge_faithful(question.question, answer, context)
        if faithful is not None:
            metrics["faithfulness"] = float(faithful)
    return {"fact_verdicts": verdicts, "abstains": abstains, "metrics": metrics}


def _generate(client: EspertoClient, dataset: Dataset, question: Question, name: str) -> dict:
    """Risposta + contesto per una configurazione (closed_book: nessun contesto)."""
    if name == CLOSED_BOOK:
        prompt = f"Rispondi in italiano a questa domanda sul progetto Python '{dataset.name}':\n{question.question}"
        return {"answer": ollama.chat(prompt, model=LLM_MODEL), "sources": []}
    use_planner, options = CONFIGS[name]
    return client.ask(question.question, use_planner, options)


def _generation_stage(
    client: EspertoClient, spec: RunSpec, dataset: Dataset, questions: list[Question],
    repo_files: list[str], run_dir: Path, status: _Status,
) -> None:
    status.update(stage="generation", done=0, total=len(spec.generation_configs) * len(questions))
    for name in spec.generation_configs:
        for question in questions:
            row = {"config": name, "id": question.id, "category": question.category}
            try:
                result = _generate(client, dataset, question, name)
                answer, sources = result["answer"], result["sources"]
                # La mappa del repository e' nel prompt: il giudice di fedelta' deve vederla.
                context = "\n\n---\n\n".join(
                    filter(None, [result.get("repo_map"), *(source["content"] for source in sources)])
                )
                row.update(answer=answer, sources=_compact(sources), error=result.get("error"))
                if row["error"]:  # stream interrotto: una risposta parziale non va giudicata
                    raise ValueError(row["error"])
                judged = _judge_answer(question, answer, context, has_context=name != CLOSED_BOOK)
                row.update(fact_verdicts=judged["fact_verdicts"], abstains=judged["abstains"])
                citations = citation_stats(answer, repo_files, [s["filename"] for s in sources])
                if name == CLOSED_BOOK:
                    citations["citation_in_context"] = None
                row["metrics"] = {**judged["metrics"], **citations, "answer_error": 0.0}
            except (httpx.HTTPError, KeyError, ValueError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["metrics"] = {"answer_error": 1.0}
            _append(run_dir / "generation.jsonl", row)
            status.advance()


def _latency_stage(client: EspertoClient, spec: RunSpec, questions: list[Question], run_dir: Path, status: _Status) -> None:
    configs = [name for name in spec.generation_configs if name != CLOSED_BOOK]
    step = max(1, len(questions) // spec.latency_questions)
    subset = questions[::step][: spec.latency_questions]
    status.update(stage="latency", done=0, total=len(configs) * len(subset) * spec.latency_repeats)
    for _ in range(spec.latency_warmups):  # carica BGE-M3 e il modello Ollama prima di misurare
        client.ask(subset[0].question, *CONFIGS["full"])
    for name in configs:
        use_planner, options = CONFIGS[name]
        for repeat in range(spec.latency_repeats):
            for question in subset:
                row = {"config": name, "id": question.id, "repeat": repeat}
                try:
                    result = client.ask(question.question, use_planner, options)
                    if result["error"]:
                        raise ValueError(result["error"])
                    row["metrics"] = {k: result[k] for k in ("retrieval_s", "ttft_s", "total_s") if result[k] is not None}
                except (httpx.HTTPError, KeyError, ValueError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                _append(run_dir / "latency.jsonl", row)
                status.advance()


def execute_run(run_id: str, spec: RunSpec, client: Optional[EspertoClient] = None) -> None:
    """Esegue il run e scrive tutti gli artefatti in RESULTS_DIR/<run_id>. Errori in status.json."""
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = _Status(run_dir, run_id)
    try:
        client = client or EspertoClient()
        dataset = load_dataset(spec.dataset)
        questions = dataset.questions[: spec.limit] if spec.limit else dataset.questions
        repo_dir, commit = fetch_repo(dataset)
        errors = validate_dataset(dataset, repo_dir)
        if errors:
            raise ValueError("Gold set non valido: " + "; ".join(errors[:5]))
        repo_files = list_repo_files(repo_dir)

        job = None
        if not spec.skip_index:
            status.update(stage="indexing")
            job = client.index_zip(build_zip(repo_dir, repo_files), f"{dataset.name}.zip")
        manifest = _manifest(run_id, spec, dataset, commit, job, client.health())
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if "retrieval" in spec.stages:
            _retrieval_stage(client, spec, questions, run_dir, status)
        if "generation" in spec.stages:
            _generation_stage(client, spec, dataset, questions, repo_files, run_dir, status)
        if "latency" in spec.stages:
            _latency_stage(client, spec, questions, run_dir, status)

        status.update(stage="report")
        report.write_report(run_dir, DATASETS_DIR / dataset.name / "human_labels.json")
        status.update(status="done", stage="done")
    except Exception as error:  # il thread non deve morire in silenzio: l'errore va in status.json
        logger.exception("Run %s fallito", run_id)
        status.update(status="error", error=f"{type(error).__name__}: {error}")
