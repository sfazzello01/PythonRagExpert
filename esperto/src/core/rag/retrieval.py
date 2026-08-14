"""Hybrid retrieval, parent retrieval, score deduplication, and context assembly."""

from __future__ import annotations

import asyncio
from typing import Iterable

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient  # type: ignore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore

from ..services.embedding import encode_queries
from ..services.qdrant import get_async_client, get_codebase_map, hybrid_search
from ..shared.settings import K
from .query import expand_queries


def _keep_highest_scoring(documents: Iterable[Document]) -> list[Document]:
    """Deduplicate equal chunks while retaining their best ColBERT score."""
    by_content: dict[str, Document] = {}
    for document in documents:
        previous = by_content.get(document.page_content)
        if previous is None or document.metadata["_colbert_score"] > previous.metadata["_colbert_score"]:
            by_content[document.page_content] = document
    return list(by_content.values())


class HybridRetriever:
    """Retrieve ranked codebase context from the BGE-M3 hybrid Qdrant index."""

    def __init__(self, client: AsyncQdrantClient | None = None):
        self._client = client or get_async_client()

    async def retrieve_with_sources(self, question: str) -> tuple[str, list[dict]]:
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
            final_documents.extend(same_file_documents or [document])

        ranked_documents = sorted(
            _keep_highest_scoring(final_documents),
            key=lambda document: document.metadata["_colbert_score"],
            reverse=True,
        )
        codebase_map = await get_codebase_map(self._client)
        source_documents = [
            document
            for document in ranked_documents
            if document.metadata.get("filename") != "__codebase_map__"
        ]
        if codebase_map:
            source_documents.append(codebase_map)
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