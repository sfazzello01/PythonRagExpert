"""Modelli delle richieste ricevute dal frontend."""

from pydantic import BaseModel


class ChatFrontendRequest(BaseModel):
    """Domanda inviata dalla pagina di chat, con le opzioni scelte dall'utente."""

    message: str
    use_planner: bool = True
    use_reasoning: bool = False
