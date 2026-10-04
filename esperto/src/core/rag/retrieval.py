"""Hybrid retrieval, same-file and symbol-graph enrichment, final rerank, and context budget."""

from __future__ import annotations

import asyncio
from itertools import chain
from typing import Iterable

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient  # type: ignore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore

from config import (
    K,
    RAG_CONTEXT_MAX_CHARS,
    RAG_CONTEXT_MAX_SOURCES,
    SEMANTIC_RELEVANCE_RATIO,
    SYMBOL_GRAPH_MAX_NEIGHBORS,
)

from ..services.embedding import encode_queries
from ..services.qdrant import (
    get_async_client,
    get_documents_by_chunk_ids,
    hybrid_search,
    rerank_chunk_ids,
)
from ..services.symbol_graph import SymbolGraph
from ..shared.options import RetrievalOptions
from ..shared.text import sanitize_text
from ..shared.vectors import QueryVectors

SAME_FILE_LIMIT = 4


def _keep_highest_scoring(documents: Iterable[Document]) -> list[Document]:
    """Deduplicate chunks by chunk_id while retaining their best score."""
    by_id: dict[str, Document] = {}
    for document in documents:
        previous = by_id.get(document.metadata["chunk_id"])
        if previous is None or document.metadata["_colbert_score"] > previous.metadata["_colbert_score"]:
            by_id[document.metadata["chunk_id"]] = document
    return list(by_id.values())


def _apply_context_budget(documents: Iterable[Document], max_sources: int, max_chars: int) -> list[Document]:
    """Keep ranked documents within the given source-count and character budgets."""
    selected: list[Document] = []
    total_characters = 0
    for document in documents:
        if len(selected) >= max_sources:
            break
        size = len(document.page_content)
        if selected and total_characters + size > max_chars:
            continue
        selected.append(document)
        total_characters += size
    return selected


def _to_source(document: Document, score: float | None) -> dict:
    metadata = document.metadata
    return {
        "filename": metadata.get("filename"),
        "service": metadata.get("service"),
        "chunk_type": metadata.get("chunk_type"),
        "symbol_name": metadata.get("symbol_name"),
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
        "chunk_id": metadata["chunk_id"],
        "score": score,
        "graph_relation": metadata.get("_graph_relation"),
        "content": document.page_content,
    }


class HybridRetriever:
    """Retrieve ranked codebase context from the BGE-M3 hybrid Qdrant index."""

    def __init__(self, client: AsyncQdrantClient | None = None):
        self._client = client or get_async_client()
        self._symbol_graph = SymbolGraph()

    async def _candidates(self, query_vectors: QueryVectors, options: RetrievalOptions) -> list[Document]:
        """Candidate chunks for one question: hybrid search, same-file chunks, graph neighbors."""
        # La soglia di rilevanza vale solo qui: chunk dello stesso file e rerank finale
        # servono a dare contesto e a ordinare, non a selezionare.
        documents = await hybrid_search(
            self._client, query_vectors, limit=K, options=options, min_score_ratio=SEMANTIC_RELEVANCE_RATIO
        )
        if options.use_same_file:
            filenames = dict.fromkeys(document.metadata["filename"] for document in documents)
            same_file = await asyncio.gather(
                *(
                    hybrid_search(
                        self._client,
                        query_vectors,
                        limit=SAME_FILE_LIMIT,
                        query_filter=Filter(
                            must=[FieldCondition(key="metadata.filename", match=MatchValue(value=filename))]
                        ),
                        options=options,
                    )
                    for filename in filenames
                )
            )
            documents = _keep_highest_scoring(chain(documents, *same_file))
        if not options.use_symbol_graph:
            return documents

        relations = self._symbol_graph.related_chunks(
            {document.metadata["chunk_id"] for document in documents},
            limit=SYMBOL_GRAPH_MAX_NEIGHBORS,
        )
        graph_documents = await get_documents_by_chunk_ids(self._client, list(relations))
        for document in graph_documents:
            document.metadata["_graph_relation"] = relations[document.metadata["chunk_id"]]
        return [*documents, *graph_documents]

    async def retrieve(
        self,
        questions: list[str],
        original_question: str,
        options: RetrievalOptions = RetrievalOptions(),
    ) -> list[dict]:
        """Return the ranked sources for the original question and its planner sub-questions."""
        original_question = sanitize_text(original_question).strip()
        retrieval_questions = [
            original_question,
            *(
                question
                for question in questions
                if question.casefold() != original_question.casefold()
            ),
        ]
        # Un solo passaggio BGE-M3 per tutte le domande; il primo vettore e' la domanda originale.
        query_vectors = await asyncio.to_thread(encode_queries, retrieval_questions)
        candidates = await asyncio.gather(
            *(self._candidates(vectors, options) for vectors in query_vectors)
        )
        documents: dict[str, Document] = {}
        neighbors: dict[str, Document] = {}
        for document in chain.from_iterable(candidates):
            target = neighbors if "_graph_relation" in document.metadata else documents
            target.setdefault(document.metadata["chunk_id"], document)

        if options.use_colbert:
            # Gli score delle sotto-domande non sono confrontabili: rerank sulla domanda originale.
            reranked = await rerank_chunk_ids(self._client, query_vectors[0], list(documents))
            scores = {
                document.metadata["chunk_id"]: document.metadata["_colbert_score"]
                for document in reranked
            }
        else:
            scores = {
                chunk_id: document.metadata.get("_colbert_score")
                for chunk_id, document in documents.items()
            }
        ranked = sorted(
            (document for chunk_id, document in documents.items() if chunk_id in scores),
            key=lambda document: -(scores[document.metadata["chunk_id"]] or 0.0),
        )
        # I vicini del grafo servono proprio quando non somigliano alla domanda: saltano il
        # rerank e hanno una quota riservata per ogni domanda, proporzionale ai loro posti
        # e mai oltre meta' del budget.
        max_neighbors = SYMBOL_GRAPH_MAX_NEIGHBORS * len(retrieval_questions)
        graph_share = min(max_neighbors / RAG_CONTEXT_MAX_SOURCES, 0.5)
        graph_documents = _apply_context_budget(
            (document for chunk_id, document in neighbors.items() if chunk_id not in documents),
            max_neighbors,
            int(RAG_CONTEXT_MAX_CHARS * graph_share),
        )
        ranked_documents = _apply_context_budget(
            ranked,
            RAG_CONTEXT_MAX_SOURCES - len(graph_documents),
            RAG_CONTEXT_MAX_CHARS - sum(len(document.page_content) for document in graph_documents),
        )
        return [
            *(_to_source(document, scores[document.metadata["chunk_id"]]) for document in ranked_documents),
            *(_to_source(document, None) for document in graph_documents),
        ]
