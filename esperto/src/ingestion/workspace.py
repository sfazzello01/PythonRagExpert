"""Gestione della workspace di indicizzazione e dello stato dei job."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from config import INDEX_STATE_DIR, INGESTION_WORKSPACE_DIR

logger = logging.getLogger("EspertoIngestion")

WORKSPACE_ROOT = INGESTION_WORKSPACE_DIR
CURRENT_LINK = WORKSPACE_ROOT / "current"

_STATE_LOCK = threading.Lock()
_STATE_FILE = INDEX_STATE_DIR / "index_jobs.json"

JOB_QUEUED = "queued"
JOB_INDEXING = "indexing"
JOB_READY = "ready"
JOB_ERROR = "error"

_current_job: dict | None = None


def _load_job() -> dict | None:
    """Carica lo stato del job corrente, dalla memoria o dal file di stato.

    Un job rimasto in coda o in corso da un avvio precedente viene marcato come fallito.

    Returns:
        Il job corrente, oppure None se non ne è mai stato avviato uno.
    """
    global _current_job
    if _current_job is not None:
        return _current_job
    if _STATE_FILE.exists():
        try:
            _current_job = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception as error:
            logger.warning("Stato job non leggibile: %s", error)
            _current_job = None
    # Un job rimasto in queued/indexing da un run precedente è morto con il
    # processo (il thread è daemon): va marcato come errore, altrimenti ogni
    # nuovo upload verrebbe rifiutato con 409 per sempre.
    if _current_job is not None and _current_job.get("status") in (JOB_QUEUED, JOB_INDEXING):
        _current_job = {
            **_current_job,
            "status": JOB_ERROR,
            "finished_at": time.time(),
            "error": "Job interrotto dal riavvio del servizio.",
        }
        _save_job(_current_job)
    return _current_job


def _save_job(job: dict) -> None:
    """Rende corrente un job e ne salva lo stato su file.

    Args:
        job: Stato del job.
    """
    global _current_job
    _current_job = job
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(job, indent=2), encoding="utf-8")


def current_job_status() -> dict:
    """Restituisce lo stato del job corrente.

    Returns:
        Lo stato del job e i suoi dettagli; `not_indexed` senza dettagli se non esiste alcun job.
    """
    with _STATE_LOCK:
        job = _load_job()
        if job is None:
            return {"status": "not_indexed", "job": None}
        return {"status": job["status"], "job": job}


def extract_to_workspace(content: bytes, names: list[str]) -> Path:
    """Estrae i file validati in una cartella di staging della workspace.

    Args:
        content: Contenuto binario dello ZIP.
        names: Nomi dei file già validati da `zip_validation.validate_zip`.

    Returns:
        La cartella di staging creata.
    """
    staging = WORKSPACE_ROOT / f"job-{uuid.uuid4().hex[:12]}"
    staging.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        allowed = set(names)
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name not in allowed:
                continue
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    logger.info("Estratti %d file in %s", len(names), staging)
    return staging


def _activate(staging: Path) -> None:
    """Attiva atomicamente la cartella di staging e rimuove le estrazioni precedenti.

    Args:
        staging: Cartella della codebase appena indicizzata.
    """
    tmp_link = WORKSPACE_ROOT / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(staging)
    os.replace(tmp_link, CURRENT_LINK)
    for previous in WORKSPACE_ROOT.glob("job-*"):
        if previous != staging:
            shutil.rmtree(previous, ignore_errors=True)


def _run_indexer_subprocess(staging: Path) -> tuple[bool, str]:
    """Lancia la pipeline di indicizzazione in un processo separato.

    La configurazione dell'indicizzazione è risolta all'import: per cambiare BASE_DIR serve un processo
    nuovo.

    Args:
        staging: Cartella della codebase da indicizzare.

    Returns:
        L'esito (True se il processo è terminato senza errori) e le ultime righe del suo log.
    """
    env = os.environ.copy()
    env["BASE_DIR"] = str(staging)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.core.indexing.pipeline"],
            env=env,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        return False, f"Avvio indicizzatore fallito: {error}"

    tail = (result.stderr or result.stdout or "")[-4000:]
    return result.returncode == 0, tail


def start_index_job(content: bytes, names: list[str], filename: str) -> dict:
    """Registra e avvia in background un job di indicizzazione per lo ZIP caricato.

    Args:
        content: Contenuto binario dello ZIP.
        names: Nomi dei file validati da estrarre.
        filename: Nome originale dell'archivio.

    Returns:
        Il job appena registrato, nello stato `queued`.

    Raises:
        RuntimeError: Se un job è già in coda o in corso.
    """
    with _STATE_LOCK:
        existing = _load_job()
        if existing is not None and existing["status"] in (JOB_QUEUED, JOB_INDEXING):
            raise RuntimeError(f"Indicizzazione già in corso (job {existing['job_id']})")

        job = {
            "job_id": uuid.uuid4().hex[:12],
            "filename": filename,
            "status": JOB_QUEUED,
            "files": len(names),
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
        }
        _save_job(job)

    threading.Thread(target=_execute_job, args=(job["job_id"], content, names), daemon=True).start()
    return job


def _update_job(job_id: str, **fields) -> None:
    """Aggiorna i campi del job corrente, se è ancora quello indicato.

    Args:
        job_id: ID del job da aggiornare.
        **fields: Campi da modificare.
    """
    with _STATE_LOCK:
        job = _load_job()
        if job is None or job["job_id"] != job_id:
            return
        job.update(fields)
        _save_job(job)


def _execute_job(job_id: str, content: bytes, names: list[str]) -> None:
    """Esegue un job: estrae i file, lancia l'indicizzazione e ne registra l'esito.

    Args:
        job_id: ID del job.
        content: Contenuto binario dello ZIP.
        names: Nomi dei file validati da estrarre.
    """
    staging: Path | None = None
    try:
        _update_job(job_id, status=JOB_INDEXING)

        staging = extract_to_workspace(content, names)

        index_start = time.time()
        success, log_tail = _run_indexer_subprocess(staging)
        indexing_seconds = round(time.time() - index_start, 2)
        logger.info("Job %s: indicizzazione durata %.2fs.", job_id, indexing_seconds)
        if success:
            _activate(staging)
            _update_job(
                job_id,
                status=JOB_READY,
                finished_at=time.time(),
                error=None,
                indexing_seconds=indexing_seconds,
            )
            logger.info("Job %s completato.", job_id)
        else:
            _update_job(
                job_id,
                status=JOB_ERROR,
                finished_at=time.time(),
                error=log_tail[-1000:],
                indexing_seconds=indexing_seconds,
            )
            logger.error("Job %s fallito: %s", job_id, log_tail[-500:])
    except Exception as error:
        logger.exception("Job %s fallito con eccezione.", job_id)
        _update_job(job_id, status=JOB_ERROR, finished_at=time.time(), error=str(error))
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
