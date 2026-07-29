"""BGE-M3 model lifecycle and encoding operations."""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from FlagEmbedding import BGEM3FlagModel  # type: ignore
from langchain_core.documents import Document

from ..shared.settings import (
    BGE_M3_DEVICE,
    BGE_M3_USE_FP16,
    DENSE_MODEL_NAME,
    EMBEDDING_MAX_LENGTH,
)
from ..shared.vectors import QueryVectors, lexical_weights_to_sparse_vector

logger = logging.getLogger("EspertoEmbedding")

_query_model: BGEM3FlagModel | None = None


def _indexing_device() -> str:
    """Return the configured indexing device, falling back safely to CPU."""
    if BGE_M3_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA richiesta ma non disponibile: uso CPU per BGE-M3.")
        return "cpu"
    return BGE_M3_DEVICE


def load_indexing_model() -> BGEM3FlagModel:
    """Create the BGE-M3 model used by the one-shot indexing process."""
    device = _indexing_device()
    use_fp16 = BGE_M3_USE_FP16 and device.startswith("cuda")
    logger.info(
        "Caricamento BGE-M3 per indexing (%s, device=%s, fp16=%s)",
        DENSE_MODEL_NAME,
        device,
        use_fp16,
    )
    return BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=use_fp16, devices=device)


def get_query_model() -> BGEM3FlagModel:
    """Lazily load BGE-M3 on CPU to leave the GPU available for Ollama."""
    global _query_model
    if _query_model is None:
        logger.info("Caricamento BGE-M3 per query su CPU.")
        _query_model = BGEM3FlagModel(DENSE_MODEL_NAME, use_fp16=False, devices="cpu")
    return _query_model


def encode_documents(model: BGEM3FlagModel, documents: Sequence[Document]) -> dict:
    """Encode an indexing batch with all BGE-M3 vector representations."""
    texts = [document.page_content for document in documents]
    return model.encode(
        texts,
        batch_size=len(texts),
        max_length=EMBEDDING_MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )


def encode_queries(texts: Sequence[str]) -> list[QueryVectors]:
    """Encode all expanded queries in one CPU BGE-M3 forward pass."""
    output = get_query_model().encode(
        list(texts),
        batch_size=len(texts),
        max_length=EMBEDDING_MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    return [
        QueryVectors(
            dense=dense_vector.tolist(),
            sparse=lexical_weights_to_sparse_vector(lexical_weights),
            colbert=colbert_vector.tolist(),
        )
        for dense_vector, lexical_weights, colbert_vector in zip(
            output["dense_vecs"], output["lexical_weights"], output["colbert_vecs"]
        )
    ]