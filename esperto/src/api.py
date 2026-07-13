from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel
from qdrant_client import QdrantClient  # type: ignore

from .rag import stream_expert_response
from config import QDRANT_HOST, QDRANT_COLLECTION

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health_check():
    """
    Verifica lo stato del servizio esperto e dell'indicizzazione.
    
    Returns:
        Dict con stato del servizio e dell'indicizzazione.
    """
    try:
        from pathlib import Path
        BASE_DIR = Path("/app")
        flag_file = BASE_DIR / ".index_complete"
        
        # Verifica se il file flag esiste (indicizzazione completata)
        if not flag_file.exists():
            return {
                "status": "indexing",
                "message": "Indicizzazione in corso"
            }
        
        # Verifica anche che Qdrant sia accessibile e abbia dati
        client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
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
    return StreamingResponse(
        stream_expert_response(request.message),
        media_type="text/plain; charset=utf-8",
    )
