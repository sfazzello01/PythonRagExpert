"""Interruttori per richiesta degli stadi di retrieval; i valori predefiniti riproducono la pipeline completa."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievalOptions(BaseModel):
    """Disattiva singoli stadi del retrieval (usato dalle ablation del benchmark)."""

    model_config = ConfigDict(frozen=True)

    use_same_file: bool = True
    use_symbol_graph: bool = True
    use_symbol_lookup: bool = True  # chunk dei simboli nominati nella domanda
    use_repo_map: bool = True
    k: int | None = Field(default=None, ge=1, le=50)  # None = K di configurazione
