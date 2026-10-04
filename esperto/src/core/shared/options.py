"""Per-request switches for retrieval stages; defaults reproduce the production pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class RetrievalOptions(BaseModel):
    """Disable individual retrieval stages (used by the benchmark ablations)."""

    model_config = ConfigDict(frozen=True)

    use_dense: bool = True
    use_sparse: bool = True
    use_colbert: bool = True
    use_same_file: bool = True
    use_symbol_graph: bool = True

    @model_validator(mode="after")
    def _require_a_first_stage(self) -> "RetrievalOptions":
        if not (self.use_dense or self.use_sparse):
            raise ValueError("Almeno uno tra use_dense e use_sparse deve essere attivo.")
        return self
