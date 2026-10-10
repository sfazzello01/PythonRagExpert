"""API del benchmark: avvio dei run, stato, verdetti, report, confronto e validazione dei dataset."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import PlainTextResponse  # type: ignore
from pydantic import BaseModel

from config import RESULTS_DIR

from . import report
from .dataset import check_name, fetch_repo, load_dataset, validate_dataset
from .runner import RunSpec, check_spec, execute_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Esperto Benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_lock = threading.Lock()
_active_run: str | None = None


def _run_dir(run_id: str) -> Path:
    """Risolve la cartella dei risultati di un run.

    Args:
        run_id: ID del run.

    Returns:
        La cartella del run.

    Raises:
        HTTPException: Con stato 404 se l'ID non è valido o il run non esiste.
    """
    if not _RUN_ID_RE.fullmatch(run_id) or not (RESULTS_DIR / run_id).is_dir():
        raise HTTPException(status_code=404, detail="Run non trovato")
    return RESULTS_DIR / run_id


def _execute_and_release(run_id: str, spec: RunSpec) -> None:
    """Esegue un run e, al termine, libera il posto del run attivo.

    Args:
        run_id: ID del run.
        spec: Specifica del run.
    """
    global _active_run
    try:
        execute_run(run_id, spec)
    finally:
        with _lock:
            _active_run = None


@app.get("/health")
def health() -> dict:
    """Verifica che il servizio di benchmark sia attivo.

    Returns:
        Lo stato `ok`.
    """
    return {"status": "ok"}


@app.post("/runs", status_code=202)
def start_run(spec: RunSpec) -> dict:
    """Avvia un nuovo run in background.

    Args:
        spec: Specifica del run: dataset, stadi e configurazioni.

    Returns:
        L'ID assegnato al run.

    Raises:
        HTTPException: Con stato 400 se la specifica non è valida, 409 se un altro run è in corso.
    """
    global _active_run
    try:
        check_name(spec.dataset)
        check_spec(spec)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _lock:
        if _active_run is not None:
            raise HTTPException(status_code=409, detail=f"Run {_active_run} in corso")
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{spec.dataset}-{uuid.uuid4().hex[:4]}"
        _active_run = run_id
    threading.Thread(target=_execute_and_release, args=(run_id, spec), daemon=True).start()
    return {"run_id": run_id}


@app.post("/runs/{run_id}/resume", status_code=202)
def resume_run(run_id: str) -> dict:
    """Riprende un run interrotto con la stessa specifica.

    L'indicizzazione e le domande già completate vengono saltate.

    Args:
        run_id: ID del run da riprendere.

    Returns:
        L'ID del run.

    Raises:
        HTTPException: Con stato 404 se il run non esiste, 409 se si è fermato prima di iniziare o se un
            altro run è in corso.
    """
    global _active_run
    manifest = _run_dir(run_id) / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=409, detail="Run fermato prima di iniziare: avviarne uno nuovo")
    spec = RunSpec(**{**json.loads(manifest.read_text(encoding="utf-8"))["spec"], "skip_index": True})
    with _lock:
        if _active_run is not None:
            raise HTTPException(status_code=409, detail=f"Run {_active_run} in corso")
        _active_run = run_id
    threading.Thread(target=_execute_and_release, args=(run_id, spec), daemon=True).start()
    return {"run_id": run_id}


class Judgment(BaseModel):
    """Verdetto esterno sulla risposta a una domanda."""

    key: str  # chiave della risposta in to_judge.jsonl
    fact_verdicts: list[bool] = []  # un verdetto per fatto chiave, nello stesso ordine
    abstains: bool  # la risposta dichiara che l'informazione non è nel codice


@app.post("/runs/{run_id}/judgments")
def submit_judgments(run_id: str, judgments: list[Judgment]) -> dict:
    """Riceve i verdetti sulle risposte di `to_judge.jsonl` e rigenera il report.

    Args:
        run_id: ID del run giudicato.
        judgments: Un verdetto per risposta.

    Returns:
        Il numero di verdetti ricevuti.

    Raises:
        HTTPException: Con stato 404 se il run non esiste, 409 se non ha risposte da giudicare.
    """
    run_dir = _run_dir(run_id)
    if not (run_dir / "to_judge.jsonl").exists():
        raise HTTPException(status_code=409, detail="Nessuna risposta da giudicare in questo run")
    (run_dir / "judgments.jsonl").write_text(
        "".join(json.dumps(judgment.model_dump()) + "\n" for judgment in judgments), encoding="utf-8"
    )
    report.write_report(run_dir)
    return {"received": len(judgments)}


@app.get("/runs")
def list_runs() -> list[str]:
    """Elenca i run presenti nella cartella dei risultati.

    Returns:
        Gli ID dei run, in ordine alfabetico.
    """
    return sorted(path.name for path in RESULTS_DIR.iterdir() if path.is_dir() and not path.name.startswith("."))


@app.get("/runs/{run_id}")
def run_status(run_id: str) -> dict:
    """Restituisce lo stato di avanzamento di un run.

    Args:
        run_id: ID del run.

    Returns:
        Il contenuto di `status.json`: stato, stadio, domande completate ed eventuale errore.

    Raises:
        HTTPException: Con stato 404 se il run non esiste.
    """
    return json.loads((_run_dir(run_id) / "status.json").read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def run_report(run_id: str) -> str:
    """Restituisce il report Markdown di un run.

    Args:
        run_id: ID del run.

    Returns:
        Il testo di `report.md`.

    Raises:
        HTTPException: Con stato 404 se il run non esiste o il report non è ancora disponibile.
    """
    report_file = _run_dir(run_id) / "report.md"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report non ancora disponibile")
    return report_file.read_text(encoding="utf-8")


@app.get("/runs/{run_id}/compare/{other_id}", response_class=PlainTextResponse)
def compare(run_id: str, other_id: str) -> str:
    """Confronta due run sulle configurazioni e sulle domande che hanno in comune.

    Args:
        run_id: ID del run di cui misurare le differenze.
        other_id: ID del run di riferimento.

    Returns:
        La tabella Markdown delle differenze appaiate (questo run meno l'altro).

    Raises:
        HTTPException: Con stato 404 se uno dei due run non esiste.
    """
    return report.compare_runs(_run_dir(run_id), _run_dir(other_id))


@app.post("/datasets/{name}/validate")
def validate(name: str) -> dict:
    """Verifica che file e simboli del gold set esistano nel checkout del repository.

    Args:
        name: Nome del dataset.

    Returns:
        Il commit del checkout, il numero di domande e gli errori trovati.

    Raises:
        HTTPException: Con stato 400 se il dataset non esiste o non è valido.
    """
    try:
        dataset = load_dataset(name)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    repo_dir, commit = fetch_repo(dataset)
    return {"commit": commit, "questions": len(dataset.questions), "errors": validate_dataset(dataset, repo_dir)}
