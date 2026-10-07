import asyncio
import logging

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel, Field

from config import (
    INDEX_COMPLETE_FILE,
    QDRANT_COLLECTION,
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    LLM_INFERENCE_TIMEOUT,
)
from .core.rag.chain import retrieve_only, stream_expert_response
from .core.services.qdrant import get_sync_client
from .core.shared.options import RetrievalOptions

logger = logging.getLogger("EspertoAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="User question to the expert",
    )
    use_planner: bool = Field(default=True, description="Use the LLM retrieval planner")
    use_reasoning: bool = Field(default=False, description="Stream the model reasoning before the answer")
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions, description="Retrieval stage switches (ablation)")


class RetrieveRequest(BaseModel):
    message: str = Field(..., min_length=MIN_MESSAGE_LENGTH, max_length=MAX_MESSAGE_LENGTH)
    use_planner: bool = True
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    subquestions: list[str] | None = Field(default=None, description="Piano gia' calcolato: salta il planner")


@app.get("/health")
def health_check():
    """
    Verifica lo stato del servizio esperto e dell'indicizzazione.
    
    Returns:
        Dict con stato del servizio e dell'indicizzazione.
    """
    try:
        # Verifica se il file flag esiste (indicizzazione completata)
        if not INDEX_COMPLETE_FILE.exists():
            return {
                "status": "not_indexed",
                "message": "Indice non presente: caricare uno ZIP tramite il servizio di ingestion"
            }
        
        # Verifica anche che Qdrant sia accessibile e abbia dati
        client = get_sync_client()
        collection_info = client.get_collection(QDRANT_COLLECTION)
        
        if collection_info.points_count == 0:
            return {
                "status": "indexing",
                "message": "Indicizzazione in corso - nessun punto in Qdrant"
            }
        
        return {
            "status": "ok",
            "message": "Indicizzazione completata",
            "points_count": collection_info.points_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Errore health check: {str(e)}")


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> dict:
    """Planner + retrieval senza generazione: restituisce sotto-domande e fonti."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Il messaggio non puo' essere vuoto.")
    subquestions, sources = await retrieve_only(
        message, request.use_planner, request.retrieval, request.subquestions
    )
    return {"subquestions": subquestions, "sources": sources}


@app.post("/ask")
async def ask_expert(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint per porre una domanda all'esperto e ricevere la risposta in streaming.

    Args:
        request (ChatRequest): La richiesta contenente la domanda dell'utente.

    Returns:
        StreamingResponse: La risposta dell'esperto in streaming, chunk per chunk.
    """
    # Il controllo pydantic sui limiti di lunghezza avviene prima dello strip:
    # qui resta solo il caso di un messaggio di soli spazi.
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Il messaggio non puo' essere vuoto.")

    logger.info(f"Query received: {len(message)} chars")
    
    async def timeout_wrapper():
        """Wrapper with an overall deadline to prevent indefinite streaming."""
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
            logger.error(f"Timeout during streaming (limit {LLM_INFERENCE_TIMEOUT}s)")
            yield "[ERROR] La risposta ha superato il timeout massimo."
        except Exception as e:
            logger.error(f"Error during streaming: {type(e).__name__}: {str(e)}")
            yield f"[ERROR] Si è verificato un errore: {type(e).__name__}"
    
    return StreamingResponse(
        timeout_wrapper(),
        media_type="text/plain; charset=utf-8",
    )
