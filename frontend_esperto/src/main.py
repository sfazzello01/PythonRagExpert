from __future__ import annotations

from typing import Any

import requests  # type: ignore
from requests.exceptions import ChunkedEncodingError  # type: ignore
from fastapi import FastAPI, Request, UploadFile, File  # type: ignore
from fastapi.responses import (  # type: ignore
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore

from .logcon import get_logger

from config import (
    ESPERTO_INDEXER_URL,
    ESPERTO_URL,
    FRONTEND_STATIC_DIR,
    FRONTEND_TEMPLATES_DIR,
)

from .classi import ChatFrontendRequest

app = FastAPI(title="Frontend Esperto")

template = Jinja2Templates(directory=str(FRONTEND_TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

logger = get_logger(__name__)


def fetch_esperto_status() -> dict[str, Any]:
    """Interroga /health dell'esperto; non solleva eccezioni."""
    try:
        url = f"{str(ESPERTO_URL).rstrip('/')}/health"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            return {
                "status": str(data.get("status", "error")),
                "points_count": data.get("points_count"),
            }
        return {"status": "error", "points_count": None}
    except Exception:
        return {"status": "error", "points_count": None}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse(
        request,
        "esperto.html",
        {"request": request, "status": fetch_esperto_status()},
    )


@app.get("/api/status")
def api_status():
    """Stato aggregato dello stack esperto (API + indexer)."""
    esperto = fetch_esperto_status()

    indexer: dict[str, Any] = {"status": "error", "job": None}
    try:
        url = f"{str(ESPERTO_INDEXER_URL).rstrip('/')}/status"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            indexer = {
                "status": str(data.get("status", "error")),
                "job": data.get("job"),
            }
    except Exception:
        pass

    return JSONResponse(content={"esperto": esperto, "indexer": indexer})


@app.post("/api/index/upload")
def upload_index(file: UploadFile = File(...)):
    """Inoltra lo ZIP caricato dall'utente all'indexer dell'esperto.

    Endpoint sincrono: FastAPI lo esegue nel threadpool, cosi' la chiamata
    bloccante a requests non ferma l'event loop durante l'upload.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        return JSONResponse(
            status_code=400,
            content={"error": "Il file deve essere un archivio .zip"},
        )

    content = file.file.read()
    try:
        url = f"{str(ESPERTO_INDEXER_URL).rstrip('/')}/upload"
        response = requests.post(
            url,
            files={"file": (filename, content, "application/zip")},
            timeout=(10, 120),
        )
    except Exception as e:
        logger.error(f"Errore di connessione all'indexer: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": f"Errore di connessione all'indexer: {e}"},
        )

    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}

    return JSONResponse(status_code=response.status_code, content=body)


@app.post("/ask")
def ask_esperto(request: ChatFrontendRequest):
    """
    Proxy streaming verso il servizio esperto.

    Il browser chiama il frontend, mentre il container frontend raggiunge
    l'esperto tramite la rete interna Docker.
    """
    message = request.message.strip()

    if not message:
        return PlainTextResponse(
            "Inserisci una domanda prima di inviare.",
            status_code=400,
        )

    try:
        esperto_url = str(ESPERTO_URL).rstrip("/")
        full_url = f"{esperto_url}/ask"
        response = requests.post(
            full_url,
            json={
                "message": message,
                "use_planner": request.use_planner,
                "use_reasoning": request.use_reasoning,
            },
            timeout=(10, 300),
            stream=True,
        )
    except Exception as e:
        logger.error(f"Errore di connessione all'esperto: {e}")
        return PlainTextResponse(
            f"Errore di connessione all'esperto: {e}",
            status_code=502,
        )

    if not response.ok:
        error_text = response.text
        response.close()
        return PlainTextResponse(
            f"Errore esperto: {response.status_code} - {error_text}",
            status_code=502,
        )

    def stream_response():
        try:
            try:
                for chunk in response.iter_content(
                    chunk_size=None,
                    decode_unicode=True,
                ):
                    if chunk:
                        yield chunk
            except ChunkedEncodingError as e:
                logger.error(f"Stream esperto interrotto: {e}")
                yield (
                    "\n\nErrore: lo stream dell'esperto si e' interrotto prima "
                    "di completare la risposta."
                )
        finally:
            response.close()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
    )
