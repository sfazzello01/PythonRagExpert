"""API del servizio di ingestion: upload ZIP e stato indicizzazione."""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore

from .workspace import current_job_status, start_index_job
from .zip_validation import MAX_UPLOAD_MB, ZipValidationError, validate_zip

logger = logging.getLogger("EspertoIngestion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Esperto Indexer")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check semplice del servizio di ingestion."""
    return {"status": "ok"}


@app.get("/status")
def get_status() -> dict:
    """
    Restituisce lo stato del job di indicizzazione corrente.

    Returns:
        Dict con status (not_indexed/queued/indexing/ready/error) e dettagli job.
    """
    return current_job_status()


@app.post("/upload", status_code=202)
async def upload_zip(file: UploadFile = File(...)) -> dict:
    """
    Riceve uno ZIP della codebase, lo valida e avvia l'indicizzazione in background.

    Returns:
        Dettagli del job accettato (status queued).
    """
    filename = file.filename or "upload.zip"
    # Un byte oltre il limite basta a validate_zip per rifiutare senza caricare tutto in RAM.
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)

    try:
        names = validate_zip(content, filename)
    except ZipValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        job = start_index_job(content, names, filename)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    logger.info(
        "Upload accettato: %s (%d file, max %d MB)",
        filename,
        len(names),
        MAX_UPLOAD_MB,
    )
    return {"status": job["status"], "job": job}
