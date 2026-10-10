"""API del servizio esperto: stato dell'indice, solo retrieval e risposta in streaming."""

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel, Field

from config import (
    INDEX_COMPLETE_FILE,
    LLM_INFERENCE_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    QDRANT_COLLECTION,
)

from .core.rag.chain import retrieve_only, stream_expert_response
from .core.services.qdrant import get_sync_client
from .core.shared.options import RetrievalOptions

logger = logging.getLogger("EspertoAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()


class ChatRequest(BaseModel):
    """Domanda per l'esperto, con gli interruttori di planner, ragionamento e stadi di retrieval."""

    message: str = Field(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="Domanda dell'utente per l'esperto",
    )
    use_planner: bool = Field(default=True, description="Usa il planner LLM per scomporre la domanda")
    use_reasoning: bool = Field(default=False, description="Trasmette il ragionamento del modello prima della risposta")
    retrieval: RetrievalOptions = Field(
        default_factory=RetrievalOptions, description="Interruttori degli stadi di retrieval (ablation)"
    )


class RetrieveRequest(BaseModel):
    """Richiesta di solo retrieval (usata dal benchmark), con un eventuale piano già calcolato."""

    message: str = Field(..., min_length=MIN_MESSAGE_LENGTH, max_length=MAX_MESSAGE_LENGTH)
    use_planner: bool = True
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    subquestions: list[str] | None = Field(default=None, description="Piano già calcolato: salta il planner")


@app.get("/health")
def health_check() -> dict:
    """Verifica lo stato del servizio e dell'indice.

    Returns:
        Lo stato (`not_indexed`, `indexing` oppure `ok`), un messaggio e, se l'indice è pronto, il
        numero di punti in Qdrant.

    Raises:
        HTTPException: Con stato 503 se la verifica fallisce, ad esempio se Qdrant non risponde.
    """
    try:
        # Il file flag esiste solo a indicizzazione completata.
        if not INDEX_COMPLETE_FILE.exists():
            return {
                "status": "not_indexed",
                "message": "Indice non presente: caricare uno ZIP tramite il servizio di ingestion",
            }

        # Qdrant deve essere raggiungibile e contenere dei punti.
        client = get_sync_client()
        collection_info = client.get_collection(QDRANT_COLLECTION)

        if collection_info.points_count == 0:
            return {
                "status": "indexing",
                "message": "Indicizzazione in corso - nessun punto in Qdrant",
            }

        return {
            "status": "ok",
            "message": "Indicizzazione completata",
            "points_count": collection_info.points_count,
        }
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Errore health check: {error}") from error


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> dict:
    """Esegue planner e retrieval senza generare la risposta.

    Args:
        request: Domanda, interruttori di retrieval ed eventuale piano già calcolato.

    Returns:
        Le sotto-domande usate, le fonti del contesto e i candidati rimasti fuori dal contesto.

    Raises:
        HTTPException: Con stato 400 se il messaggio contiene solo spazi.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")
    subquestions, sources, dropped = await retrieve_only(
        message, request.use_planner, request.retrieval, request.subquestions
    )
    return {"subquestions": subquestions, "sources": sources, "dropped": dropped}


@app.post("/ask")
async def ask_expert(request: ChatRequest) -> StreamingResponse:
    """Pone la domanda all'esperto e ne trasmette la risposta in streaming.

    Args:
        request: Domanda dell'utente con gli interruttori della pipeline.

    Returns:
        Lo stream di testo con un evento JSON per riga: fonti, ragionamento e parti della risposta.

    Raises:
        HTTPException: Con stato 400 se il messaggio contiene solo spazi.
    """
    # Il controllo pydantic sui limiti di lunghezza avviene prima dello strip:
    # qui resta solo il caso di un messaggio di soli spazi.
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")

    logger.info("Domanda ricevuta: %d caratteri.", len(message))

    async def timeout_wrapper() -> AsyncIterator[str]:
        """Inoltra lo stream dell'esperto entro una scadenza complessiva.

        Yields:
            Gli eventi prodotti dall'esperto; alla scadenza o in caso di errore, una riga `[ERROR] ...`.
        """
        stream = stream_expert_response(
            message,
            use_planner=request.use_planner,
            use_reasoning=request.use_reasoning,
            options=request.retrieval,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + LLM_INFERENCE_TIMEOUT
        try:
            while True:
                yield await asyncio.wait_for(anext(stream), deadline - loop.time())
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            logger.error("Timeout durante lo streaming (limite %ss).", LLM_INFERENCE_TIMEOUT)
            yield "[ERROR] La risposta ha superato il timeout massimo."
        except Exception as error:
            logger.error("Errore durante lo streaming: %s: %s", type(error).__name__, error)
            yield f"[ERROR] Si è verificato un errore: {type(error).__name__}"

    return StreamingResponse(
        timeout_wrapper(),
        media_type="text/plain; charset=utf-8",
    )
