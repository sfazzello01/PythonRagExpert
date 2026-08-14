"""Orchestration for semantic indexing into the hybrid Qdrant collection."""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

from ..services.embedding import encode_documents, load_indexing_model
from ..services.index_state import clear, is_current, save, source_fingerprint
from ..services.qdrant import ensure_hybrid_collection, upsert_encoded_documents, wait_for_qdrant
from ..shared.settings import BATCH_SIZE, QDRANT_COLLECTION
from .chunking import PythonASTContextualChunker, fallback_text_split
from .discovery import iter_source_files
from .summary import build_codebase_summary_documents
from .symbol_graph import save_symbol_graph

logger = logging.getLogger("EspertoIndexing")


def _chunk_sources(files: list) -> list[Document]:
    """Create semantic documents for all supported source files."""
    documents: list[Document] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            if path.suffix == ".py":
                documents.extend(PythonASTContextualChunker(path, text).extract_chunks())
            else:
                documents.extend(fallback_text_split(path, text))
        except Exception as error:
            logger.error("Errore processamento %s: %s", path, error)
    return documents


def build_index() -> None:
    """Build or reuse the persistent hybrid index for the configured codebase."""
    start_time = time.time()
    client = wait_for_qdrant()
    source_files = iter_source_files()
    logger.info("Scansione completata: %d file validi.", len(source_files))
    fingerprint = source_fingerprint(source_files)

    if client.collection_exists(QDRANT_COLLECTION) and is_current(fingerprint):
        from ..shared.settings import INDEX_COMPLETE_FILE

        INDEX_COMPLETE_FILE.touch()
        logger.info("Codebase invariata: indicizzazione saltata.")
        return

    clear()
    ensure_hybrid_collection(client, recreate=client.collection_exists(QDRANT_COLLECTION))
    model = load_indexing_model()
    documents = _chunk_sources(source_files)
    documents.extend(build_codebase_summary_documents(source_files))
    save_symbol_graph(source_files)
    logger.info("Avvio indicizzazione ibrida di %d chunk semantici.", len(documents))

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        batch_start_time = time.time()
        upsert_encoded_documents(client, batch, encode_documents(model, batch))
        logger.info(
            "Indicizzati %d/%d chunk in %.1fs.",
            min(start + BATCH_SIZE, len(documents)),
            len(documents),
            time.time() - batch_start_time,
        )

    save(fingerprint, len(source_files))
    logger.info("Indicizzazione completata in %.2fs.", time.time() - start_time)