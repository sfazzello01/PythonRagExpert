"""Hybrid retrieval, parent retrieval, score deduplication, and context assembly."""

from __future__ import annotations

import asyncio
from typing import Iterable

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient  # type: ignore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore

from ..services.embedding import encode_queries
from ..services.qdrant import (
    get_async_client,
    get_codebase_map,
    get_documents_by_chunk_ids,
    hybrid_search,
    rerank_chunk_ids,
)
from ..services.symbol_graph import SymbolGraph
from ..shared.settings import (
    K,
    RAG_CONTEXT_MAX_CHARS,
    RAG_CONTEXT_MAX_SOURCES,
    SYMBOL_GRAPH_MAX_NEIGHBORS,
)
from .query import expand_queries

_ARCHITECTURE_MARKERS = (
    "architettura",
    "architecture",
    "panoramica",
    "struttura del progetto",
    "struttura progetto",
    "quali servizi",
    "intero progetto",
    "intera codebase",
)


def _keep_highest_scoring(documents: Iterable[Document]) -> list[Document]:
    """Deduplicate equal chunks while retaining their best ColBERT score."""
    by_content: dict[str, Document] = {}
    for document in documents:
        previous = by_content.get(document.page_content)
        if previous is None or document.metadata["_colbert_score"] > previous.metadata["_colbert_score"]:
            by_content[document.page_content] = document
    return list(by_content.values())


def _is_architectural_question(question: str) -> bool:
    normalized = question.casefold()
    return any(marker in normalized for marker in _ARCHITECTURE_MARKERS)


def _document_sort_key(document: Document) -> tuple[int, float]:
    return (
        document.metadata.get("_context_priority", 0),
        -document.metadata.get("_colbert_score", 0.0),
    )


def _limit_context_documents(documents: Iterable[Document]) -> list[Document]:
    """Keep the most direct evidence within a stable source and character budget."""
    selected: list[Document] = []
    total_characters = 0
    for document in sorted(documents, key=_document_sort_key):
        if len(selected) >= RAG_CONTEXT_MAX_SOURCES:
            break
        document_size = len(document.page_content)
        if selected and total_characters + document_size > RAG_CONTEXT_MAX_CHARS:
            continue
        selected.append(document)
        total_characters += document_size
    return selected


class HybridRetriever:
    """Retrieve ranked codebase context from the BGE-M3 hybrid Qdrant index."""

    def __init__(self, client: AsyncQdrantClient | None = None):
        self._client = client or get_async_client()
        self._symbol_graph = SymbolGraph()

    async def retrieve_with_sources(
        self,
        question: str,
        include_codebase_map: bool = False,
    ) -> tuple[str, list[dict]]:
        """Build de-duplicated context and return the ranked source chunks."""
        queries = expand_queries(question)
        if not queries:
            return "", []

        query_vectors = await asyncio.to_thread(encode_queries, queries)
        original_query_vectors = query_vectors[0]
        initial_results = await asyncio.gather(
            *(hybrid_search(self._client, query_vectors, limit=K) for query_vectors in query_vectors)
        )
        unique_documents = _keep_highest_scoring(
            document for results in initial_results for document in results
        )
        primary_contents = {document.page_content for document in unique_documents}
        for document in unique_documents:
            document.metadata["_context_priority"] = 0

        final_documents: list[Document] = []
        processed_files: set[str] = set()
        for document in unique_documents:
            filename = document.metadata.get("filename")
            if not filename or filename == "__codebase_map__":
                final_documents.append(document)
                continue
            if filename in processed_files:
                continue

            processed_files.add(filename)
            same_file_documents = await hybrid_search(
                self._client,
                original_query_vectors,
                limit=4,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.filename",
                            match=MatchValue(value=filename),
                        )
                    ]
                ),
            )
            for same_file_document in same_file_documents:
                same_file_document.metadata["_context_priority"] = (
                    0 if same_file_document.page_content in primary_contents else 1
                )
            final_documents.extend(same_file_documents or [document])

        ranked_documents = sorted(_keep_highest_scoring(final_documents), key=_document_sort_key)
        graph_relations = self._symbol_graph.related_chunks(
            {
                chunk_id
                for document in ranked_documents
                if (chunk_id := document.metadata.get("chunk_id"))
            },
            limit=SYMBOL_GRAPH_MAX_NEIGHBORS,
        )
        graph_documents = await get_documents_by_chunk_ids(
            self._client,
            list(graph_relations),
        )
        for document in graph_documents:
            document.metadata["_graph_relation"] = graph_relations[
                document.metadata["chunk_id"]
            ]
            document.metadata["_context_priority"] = 2
        source_documents = [
            document
            for document in ranked_documents
            if document.metadata.get("filename") != "__codebase_map__"
        ]
        source_documents.extend(graph_documents)
        if include_codebase_map or _is_architectural_question(question):
            codebase_map = await get_codebase_map(self._client)
            if codebase_map:
                codebase_map.metadata["_context_priority"] = 3
                source_documents.append(codebase_map)
        source_documents = _limit_context_documents(source_documents)
        sources = [
            {
                "filename": document.metadata.get("filename"),
                "service": document.metadata.get("service"),
                "chunk_type": document.metadata.get("chunk_type"),
                "symbol_name": document.metadata.get("symbol_name"),
                "start_line": document.metadata.get("start_line"),
                "end_line": document.metadata.get("end_line"),
                "chunk_id": document.metadata.get("chunk_id"),
                "score": document.metadata.get("_colbert_score"),
                "_context_priority": document.metadata.get("_context_priority", 0),
                "graph_relation": document.metadata.get("_graph_relation"),
                "content": document.page_content,
            }
            for document in source_documents
            if document.metadata.get("filename")
        ]
        context_str = "\n\n---\n\n".join(document.page_content for document in source_documents)
        return context_str, sources

    async def retrieve_context(self, question: str) -> str:
        """Build a de-duplicated context ordered by descending ColBERT relevance."""
        context, _ = await self.retrieve_with_sources(question)
        return context

    async def retrieve_many_with_sources(
        self,
        questions: list[str],
        original_question: str,
        include_codebase_map: bool = False,
    ) -> tuple[str, list[dict]]:
        """Merge planner contexts, rerank candidates, and apply one shared budget."""
        results = await asyncio.gather(
            *(
                self.retrieve_with_sources(
                    question,
                    include_codebase_map=include_codebase_map,
                )
                for question in questions[:2]
            )
        )
        sources_by_key: dict[tuple[str | None, str | None], dict] = {}
        for _, sources in results:
            for source in sources:
                key = (source.get("filename"), source.get("chunk_id"))
                sources_by_key.setdefault(key, source)
        candidate_chunk_ids = [
            chunk_id
            for source in sources_by_key.values()
            if (chunk_id := source.get("chunk_id"))
        ]
        original_vectors = await asyncio.to_thread(encode_queries, [original_question])
        reranked_documents = await rerank_chunk_ids(
            self._client,
            original_vectors[0],
            candidate_chunk_ids,
        )
        rerank_scores = {
            document.metadata["chunk_id"]: document.metadata["_colbert_score"]
            for document in reranked_documents
        }
        sources = list(sources_by_key.values())
        if rerank_scores:
            sources.sort(
                key=lambda source: -float(rerank_scores.get(source.get("chunk_id"), -1.0)),
            )
        else:
            sources.sort(
                key=lambda source: (
                    source.get("_context_priority", 0),
                    -float(source.get("score") or 0.0),
                ),
            )
        for source in sources:
            source["final_score"] = rerank_scores.get(source.get("chunk_id"))
        selected_sources: list[dict] = []
        total_characters = 0
        for source in sources:
            if len(selected_sources) >= RAG_CONTEXT_MAX_SOURCES:
                break
            source_size = len(source["content"])
            if selected_sources and total_characters + source_size > RAG_CONTEXT_MAX_CHARS:
                continue
            selected_sources.append(source)
            total_characters += source_size
        sources = selected_sources
        return "\n\n---\n\n".join(source["content"] for source in sources), sources