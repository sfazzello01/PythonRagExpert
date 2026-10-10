"""Frontend web dell'esperto: pagina di chat, stato dell'indice e proxy verso i servizi interni."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests  # type: ignore
from fastapi import FastAPI, File, Request, UploadFile  # type: ignore
from fastapi.responses import (  # type: ignore
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from requests.exceptions import ChunkedEncodingError  # type: ignore

from config import (
    ESPERTO_INDEXER_URL,
    ESPERTO_URL,
    FRONTEND_STATIC_DIR,
    FRONTEND_TEMPLATES_DIR,
)

from .classi import ChatFrontendRequest
from .logcon import get_logger

app = FastAPI(title="Frontend Esperto")

template = Jinja2Templates(directory=str(FRONTEND_TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

logger = get_logger(__name__)


def fetch_esperto_status() -> dict[str, Any]:
    """Interroga `/health` dell'esperto senza sollevare eccezioni.

    Returns:
        Lo stato dell'esperto e il numero di punti dell'indice; lo stato è `error` se la richiesta
        fallisce.
    """
    try:
        response = requests.get(f"{ESPERTO_URL}/health", timeout=10)
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
def home(request: Request) -> HTMLResponse:
    """Mostra la pagina principale: la chat con l'esperto e lo stato dell'indice.

    Args:
        request: Richiesta HTTP, necessaria al template.

    Returns:
        La pagina HTML.
    """
    return template.TemplateResponse(
        request,
        "esperto.html",
        {"request": request, "status": fetch_esperto_status()},
    )


@app.get("/api/status")
def api_status() -> JSONResponse:
    """Restituisce lo stato aggregato dello stack: API dell'esperto e indexer.

    Returns:
        La risposta JSON con le chiavi `esperto` e `indexer`.
    """
    esperto = fetch_esperto_status()

    indexer: dict[str, Any] = {"status": "error", "job": None}
    try:
        response = requests.get(f"{ESPERTO_INDEXER_URL}/status", timeout=10)
        if response.ok:
            data = response.json()
            indexer = {
                "status": str(data.get("status", "error")),
                "job": data.get("job"),
            }
    except Exception:
        pass  # indexer non raggiungibile: resta lo stato di errore

    return JSONResponse(content={"esperto": esperto, "indexer": indexer})


@app.post("/api/index/upload")
def upload_index(file: UploadFile = File(...)) -> JSONResponse:
    """Inoltra lo ZIP caricato dall'utente all'indexer dell'esperto.

    Endpoint sincrono: FastAPI lo esegue nel threadpool, così la chiamata bloccante a requests non ferma
    l'event loop durante l'upload.

    Args:
        file: Archivio ZIP caricato dal browser.

    Returns:
        La risposta dell'indexer, oppure un errore 400 (il file non è uno ZIP) o 502 (indexer non
        raggiungibile).
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        return JSONResponse(
            status_code=400,
            content={"error": "Il file deve essere un archivio .zip"},
        )

    content = file.file.read()
    try:
        response = requests.post(
            f"{ESPERTO_INDEXER_URL}/upload",
            files={"file": (filename, content, "application/zip")},
            timeout=(10, 120),
        )
    except Exception as error:
        logger.error("Errore di connessione all'indexer: %s", error)
        return JSONResponse(
            status_code=502,
            content={"error": f"Errore di connessione all'indexer: {error}"},
        )

    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text}

    return JSONResponse(status_code=response.status_code, content=body)


@app.post("/ask")
def ask_esperto(request: ChatFrontendRequest) -> Response:
    """Inoltra la domanda all'esperto e ne ritrasmette la risposta in streaming.

    Il browser chiama il frontend, mentre il container del frontend raggiunge l'esperto tramite la rete
    interna di Docker.

    Args:
        request: Domanda e opzioni scelte nella pagina di chat.

    Returns:
        Lo stream della risposta, oppure un errore 400 (domanda vuota) o 502 (esperto non raggiungibile
        o in errore).
    """
    message = request.message.strip()

    if not message:
        return PlainTextResponse(
            "Inserisci una domanda prima di inviare.",
            status_code=400,
        )

    try:
        response = requests.post(
            f"{ESPERTO_URL}/ask",
            json={
                "message": message,
                "use_planner": request.use_planner,
                "use_reasoning": request.use_reasoning,
            },
            timeout=(10, 300),
            stream=True,
        )
    except Exception as error:
        logger.error("Errore di connessione all'esperto: %s", error)
        return PlainTextResponse(
            f"Errore di connessione all'esperto: {error}",
            status_code=502,
        )

    if not response.ok:
        error_text = response.text
        response.close()
        return PlainTextResponse(
            f"Errore esperto: {response.status_code} - {error_text}",
            status_code=502,
        )

    def stream_response() -> Iterator[str]:
        """Ritrasmette i blocchi della risposta e chiude sempre la connessione verso l'esperto.

        Yields:
            I blocchi di testo ricevuti; se lo stream si interrompe, un messaggio di errore finale.
        """
        try:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
        except ChunkedEncodingError as error:
            logger.error("Stream esperto interrotto: %s", error)
            yield "\n\nErrore: lo stream dell'esperto si è interrotto prima di completare la risposta."
        finally:
            response.close()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
    )
