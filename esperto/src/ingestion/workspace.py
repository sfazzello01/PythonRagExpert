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

logger = logging.getLogger("EspertoIngestion")

WORKSPACE_ROOT = Path(os.getenv("INGESTION_WORKSPACE_DIR", "/app/workspace"))
CURRENT_LINK = WORKSPACE_ROOT / "current"
UPLOADS_DIR = WORKSPACE_ROOT / "uploads"

_STATE_LOCK = threading.Lock()
_STATE_FILE: Path | None = None

JOB_QUEUED = "queued"
JOB_INDEXING = "indexing"
JOB_READY = "ready"
JOB_ERROR = "error"

_current_job: dict | None = None


def _state_file() -> Path:
    global _STATE_FILE
    if _STATE_FILE is None:
        state_dir = Path(os.getenv("INDEX_STATE_DIR", "/app/state"))
        _STATE_FILE = state_dir / "index_jobs.json"
    return _STATE_FILE


def _load_job() -> dict | None:
    global _current_job
    if _current_job is not None:
        return _current_job
    state_file = _state_file()
    if state_file.exists():
        try:
            _current_job = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Stato job non leggibile: %s", e)
            _current_job = None
    # Un job rimasto in queued/indexing da un run precedente e' morto con il
    # processo (il thread e' daemon): va marcato come errore, altrimenti ogni
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
    global _current_job
    _current_job = job
    state_file = _state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(job, indent=2), encoding="utf-8")


def current_job_status() -> dict:
    """Restituisce lo stato del job corrente (o not_indexed se assente)."""
    with _STATE_LOCK:
        job = _load_job()
        if job is None:
            return {"status": "not_indexed", "job": None}
        return {"status": job["status"], "job": job}


def extract_to_workspace(content: bytes, names: list[str]) -> Path:
    """
    Estrae i file validati in una directory di staging sotto la workspace.

    Args:
        content: Contenuto binario dello ZIP.
        names: Nomi dei file gia' validati da zip_validation.validate_zip.

    Returns:
        Path della directory di staging creata.
    """
    staging = WORKSPACE_ROOT / f"job-{uuid.uuid4().hex[:12]}"
    staging.mkdir(parents=True, exist_ok=False)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
    """Attiva atomicamente la directory di staging come workspace corrente."""
    tmp_link = WORKSPACE_ROOT / ".current.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(staging)
    os.replace(tmp_link, CURRENT_LINK)


def _run_indexer_subprocess(staging: Path) -> tuple[bool, str]:
    """
    Lancia la pipeline di indicizzazione come processo separato con env override.

    La configurazione dell'indicizzazione viene risolta a import-time, quindi
    l'override di BASE_DIR/INDEX_TARGET_PATHS deve avvenire in un processo nuovo.

    Returns:
        (successo, output_coda) tupla con esito e ultime righe di log.
    """
    env = os.environ.copy()
    env["BASE_DIR"] = str(staging)
    env["INDEX_TARGET_PATHS"] = "."

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.indicizzatore"],
            env=env,
            capture_output=True,
            text=True,
            timeout=None,
        )
    except Exception as e:
        return False, f"Avvio indicizzatore fallito: {e}"

    tail = (result.stderr or result.stdout or "")[-4000:]
    return result.returncode == 0, tail


def start_index_job(content: bytes, names: list[str], filename: str) -> dict:
    """
    Registra e avvia in background un job di indicizzazione per lo ZIP caricato.

    Raises:
        RuntimeError: Se un job e' gia' in corso (queued o indexing).
    """
    with _STATE_LOCK:
        existing = _load_job()
        if existing is not None and existing["status"] in (JOB_QUEUED, JOB_INDEXING):
            raise RuntimeError(
                f"Indicizzazione gia' in corso (job {existing['job_id']})"
            )

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

    thread = threading.Thread(
        target=_execute_job,
        args=(job["job_id"], content, names),
        daemon=True,
    )
    thread.start()
    return job


def _update_job(job_id: str, **fields) -> None:
    with _STATE_LOCK:
        job = _load_job()
        if job is None or job["job_id"] != job_id:
            return
        job.update(fields)
        _save_job(job)


def _execute_job(job_id: str, content: bytes, names: list[str]) -> None:
    staging: Path | None = None
    try:
        _update_job(job_id, status=JOB_INDEXING)

        staging = extract_to_workspace(content, names)

        # Punta la workspace corrente alla nuova estrazione prima di indicizzare,
        # cosi' in caso di riavvio il prossimo run vede i nuovi sorgenti.
        _activate(staging)

        index_start = time.time()
        success, log_tail = _run_indexer_subprocess(staging)
        indexing_seconds = round(time.time() - index_start, 2)
        logger.info("Job %s: indicizzazione durata %.2fs.", job_id, indexing_seconds)
        if success:
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
    except Exception as e:
        logger.exception("Job %s fallito con eccezione.", job_id)
        _update_job(
            job_id,
            status=JOB_ERROR,
            finished_at=time.time(),
            error=str(e),
        )
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
