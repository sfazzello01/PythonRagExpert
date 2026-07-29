import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel, Field

from .rag import stream_expert_response
from config import (
    QDRANT_HOST,
    QDRANT_COLLECTION,
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    LLM_INFERENCE_TIMEOUT,
)
from .core.services.qdrant import get_sync_client

logger = logging.getLogger("EspertoAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=3, max_length=5000, description="User question to the expert")


@app.get("/health")
def health_check():
    """
    Verifica lo stato del servizio esperto e dell'indicizzazione.
    
    Returns:
        Dict con stato del servizio e dell'indicizzazione.
    """
    try:
        state_dir = Path(os.getenv("INDEX_STATE_DIR", "/app/state"))
        flag_file = state_dir / "index_complete"
        
        # Verifica se il file flag esiste (indicizzazione completata)
        if not flag_file.exists():
            return {
                "status": "indexing",
                "message": "Indicizzazione in corso"
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


@app.post("/ask")
async def ask_expert(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint per porre una domanda all'esperto e ricevere la risposta in streaming.

    Args:
        request (ChatRequest): La richiesta contenente la domanda dell'utente.

    Returns:
        StreamingResponse: La risposta dell'esperto in streaming, chunk per chunk.
    """
    # Validate message length
    message = request.message.strip()
    if len(message) < MIN_MESSAGE_LENGTH or len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message length must be between {MIN_MESSAGE_LENGTH} and {MAX_MESSAGE_LENGTH} characters"
        )
    
    logger.info(f"Query received: {len(message)} chars")
    
    async def timeout_wrapper():
        """Wrapper with timeout to prevent indefinite streaming."""
        try:
            async for chunk in stream_expert_response(message):
                yield chunk
        except asyncio.TimeoutError:
            logger.error(f"LLM inference timeout after {LLM_INFERENCE_TIMEOUT}s")
            yield "[ERROR] La risposta ha superato il timeout massimo."
        except Exception as e:
            logger.error(f"Error during streaming: {type(e).__name__}: {str(e)}")
            yield f"[ERROR] Si è verificato un errore: {type(e).__name__}"
    
    return StreamingResponse(
        timeout_wrapper(),
        media_type="text/plain; charset=utf-8",
    )
