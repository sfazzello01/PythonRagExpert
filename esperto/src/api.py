from fastapi import FastAPI  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel

from .rag import stream_expert_response

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


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
