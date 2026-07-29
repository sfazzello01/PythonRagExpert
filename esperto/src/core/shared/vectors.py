"""Qdrant vector value types and conversions shared by the RAG pipeline."""

from dataclasses import dataclass
from typing import Mapping

from qdrant_client.http.models import SparseVector  # type: ignore


@dataclass(frozen=True)
class QueryVectors:
    """Dense, sparse, and ColBERT representations of a search query."""

    dense: list[float]
    sparse: SparseVector
    colbert: list[list[float]]


def lexical_weights_to_sparse_vector(lexical_weights: Mapping[str, float]) -> SparseVector:
    """Convert BGE-M3 lexical weights to Qdrant's sparse-vector format."""
    if not lexical_weights:
        return SparseVector(indices=[], values=[])

    indices, values = zip(
        *((int(token_id), float(weight)) for token_id, weight in lexical_weights.items())
    )
    return SparseVector(indices=list(indices), values=list(values))