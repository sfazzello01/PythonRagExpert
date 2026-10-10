"""API del servizio di ingestion: upload dello ZIP e stato dell'indicizzazione."""

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
    """Verifica che il servizio di ingestion sia attivo.

    Returns:
        Lo stato `ok`.
    """
    return {"status": "ok"}


@app.get("/status")
def get_status() -> dict:
    """Restituisce lo stato del job di indicizzazione corrente.

    Returns:
        Lo stato (`not_indexed`, `queued`, `indexing`, `ready` o `error`) e i dettagli del job.
    """
    return current_job_status()


@app.post("/upload", status_code=202)
async def upload_zip(file: UploadFile = File(...)) -> dict:
    """Riceve lo ZIP della codebase, lo valida e avvia l'indicizzazione in background.

    Args:
        file: Archivio ZIP caricato.

    Returns:
        Lo stato e i dettagli del job accettato (`queued`).

    Raises:
        HTTPException: Con stato 400 se l'archivio non è valido, 409 se un'indicizzazione è già in
            corso.
    """
    filename = file.filename or "upload.zip"
    # Un byte oltre il limite basta a validate_zip per rifiutare senza caricare tutto in RAM.
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)

    try:
        names = validate_zip(content, filename)
    except ZipValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        job = start_index_job(content, names, filename)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    logger.info("Upload accettato: %s (%d file, max %d MB)", filename, len(names), MAX_UPLOAD_MB)
    return {"status": job["status"], "job": job}
