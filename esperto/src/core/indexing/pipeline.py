"""Orchestration for semantic indexing into the hybrid Qdrant collection."""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

from ..services.embedding import encode_documents, load_indexing_model
from ..services.index_state import is_current, save, source_fingerprint
from ..services.qdrant import (
    activate_collection,
    create_hybrid_collection,
    has_active_collection,
    upsert_encoded_documents,
    wait_for_qdrant,
)
from config import BATCH_SIZE, INDEX_COMPLETE_FILE, QDRANT_COLLECTION

from .chunking import PythonASTContextualChunker, fallback_text_split
from .discovery import iter_source_files
from .repo_map import save_repo_map
from .symbol_graph import build_symbol_graph, chunk_relations, save_symbol_graph

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


def _add_relation_headers(documents: list[Document], relations: dict[str, dict[str, list[str]]]) -> None:
    """Prepend to each chunk what it inherits from and calls, so the answering model reads it."""
    for document in documents:
        related = relations.get(document.metadata["chunk_id"], {})
        labels = (("EREDITA DA", related.get("inherits")), ("CHIAMA", related.get("calls")))
        header = "".join(f"# {label}: {', '.join(names)}\n" for label, names in labels if names)
        document.page_content = header + document.page_content


def build_index() -> None:
    """Build or reuse the persistent hybrid index for the configured codebase."""
    start_time = time.time()
    client = wait_for_qdrant()
    source_files = iter_source_files()
    logger.info("Scansione completata: %d file validi.", len(source_files))
    fingerprint = source_fingerprint(source_files)

    if has_active_collection(client) and is_current(fingerprint):
        INDEX_COMPLETE_FILE.touch()
        logger.info("Codebase invariata: indicizzazione saltata.")
        return

    # Il nuovo indice nasce in una collection a parte: quello attivo resta servito
    # (con il suo symbol graph e il suo stato) finche' la costruzione non e' completa.
    collection = f"{QDRANT_COLLECTION}_{int(start_time)}"
    create_hybrid_collection(client, collection)
    model = load_indexing_model()
    graph = build_symbol_graph(source_files)
    documents = _chunk_sources(source_files)
    relations = chunk_relations(graph)
    logger.info("Avvio indicizzazione ibrida di %d chunk semantici.", len(documents))

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        batch_start_time = time.time()
        # L'intestazione con le relazioni e' aggiunta dopo l'encoding: i nomi dei simboli
        # chiamati, nel testo indicizzato, farebbero somigliare un chunk alle domande su di essi.
        encoded = encode_documents(model, batch)
        _add_relation_headers(batch, relations)
        upsert_encoded_documents(client, collection, batch, encoded)
        logger.info(
            "Indicizzati %d/%d chunk in %.1fs.",
            min(start + BATCH_SIZE, len(documents)),
            len(documents),
            time.time() - batch_start_time,
        )

    activate_collection(client, collection)
    save_symbol_graph(graph)
    save_repo_map(documents, source_files)
    save(fingerprint, len(source_files))
    logger.info("Indicizzazione completata in %.2fs.", time.time() - start_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_index()
