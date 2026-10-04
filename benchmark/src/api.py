"""API del servizio di benchmark: avvio run, stato, report, validazione dataset."""

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

from config import RESULTS_DIR

from .dataset import check_name, fetch_repo, load_dataset, validate_dataset
from .runner import RunSpec, check_spec, execute_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Esperto Benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_lock = threading.Lock()
_active_run: str | None = None


def _run_dir(run_id: str) -> Path:
    if not _RUN_ID_RE.fullmatch(run_id) or not (RESULTS_DIR / run_id).is_dir():
        raise HTTPException(status_code=404, detail="Run non trovato")
    return RESULTS_DIR / run_id


def _execute_and_release(run_id: str, spec: RunSpec) -> None:
    global _active_run
    try:
        execute_run(run_id, spec)
    finally:
        with _lock:
            _active_run = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/runs", status_code=202)
def start_run(spec: RunSpec) -> dict:
    global _active_run
    try:
        check_name(spec.dataset)
        check_spec(spec)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    with _lock:
        if _active_run is not None:
            raise HTTPException(status_code=409, detail=f"Run {_active_run} in corso")
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{spec.dataset}-{uuid.uuid4().hex[:4]}"
        _active_run = run_id
    threading.Thread(target=_execute_and_release, args=(run_id, spec), daemon=True).start()
    return {"run_id": run_id}


@app.get("/runs")
def list_runs() -> list[str]:
    return sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


@app.get("/runs/{run_id}")
def run_status(run_id: str) -> dict:
    return json.loads((_run_dir(run_id) / "status.json").read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def run_report(run_id: str) -> str:
    report = _run_dir(run_id) / "report.md"
    if not report.exists():
        raise HTTPException(status_code=404, detail="Report non ancora disponibile")
    return report.read_text(encoding="utf-8")


@app.post("/datasets/{name}/validate")
def validate(name: str) -> dict:
    """Verifica che file e simboli del gold set esistano nel checkout del repo."""
    try:
        dataset = load_dataset(name)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    repo_dir, commit = fetch_repo(dataset)
    return {"commit": commit, "questions": len(dataset.questions), "errors": validate_dataset(dataset, repo_dir)}
