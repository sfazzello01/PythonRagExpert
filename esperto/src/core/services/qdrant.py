"""Qdrant collection management, ingestion, and hybrid-search primitives."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional, Sequence

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, QdrantClient  # type: ignore
from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore
from qdrant_client.http.models import (  # type: ignore
    Distance,
    Filter,
    Fusion,
    FusionQuery,
    HasIdCondition,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    Prefetch,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SparseVectorParams,
    VectorParams,
)

from ..shared.settings import (
    DENSE_DIMENSION,
    PREFETCH_LIMIT,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_QUERY_TIMEOUT,
    SEMANTIC_RELEVANCE_RATIO,
)
from ..shared.options import RetrievalOptions
from ..shared.vectors import QueryVectors, lexical_weights_to_sparse_vector

logger = logging.getLogger("EspertoQdrant")

# Candidati della fusione dense+sparse riordinati dal reranker ColBERT.
RERANK_CANDIDATE_LIMIT = 75

# Con scalar quantization attiva, rescore sui vettori originali per la qualita'.
_QUANTIZATION_PARAMS = SearchParams(
    quantization=QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0)
)


def get_sync_client() -> QdrantClient:
    """Create the synchronous client used by the indexing and health-check paths."""
    return QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")


def get_async_client() -> AsyncQdrantClient:
    """Create the asynchronous client used by request-time retrieval."""
    return AsyncQdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")


async def get_documents_by_chunk_ids(
    client: AsyncQdrantClient,
    chunk_ids: list[str],
) -> list[Document]:
    """Retrieve indexed source chunks by their stable application-level IDs."""
    if not chunk_ids:
        return []
    try:
        records = await client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=[_point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
    except Exception as error:
        logger.error("Graph chunk retrieval failed: %s: %s", type(error).__name__, error)
        return []
    documents_by_chunk_id = {
        payload.get("metadata", {}).get("chunk_id"): Document(
            page_content=payload.get("page_content", ""),
            metadata=payload.get("metadata", {}),
        )
        for record in records
        if (payload := record.payload)
    }
    return [
        documents_by_chunk_id[chunk_id]
        for chunk_id in chunk_ids
        if chunk_id in documents_by_chunk_id
    ]


def wait_for_qdrant() -> QdrantClient:
    """Wait up to sixty seconds for Qdrant to accept collection requests."""
    client = get_sync_client()
    for attempt in range(1, 31):
        try:
            client.get_collections()
            logger.info("Qdrant e online e risponde.")
            return client
        except Exception as error:
            logger.warning("Attesa Qdrant (%d/30): %s", attempt, error)
            time.sleep(2)
    raise RuntimeError("Qdrant non disponibile dopo 60 secondi.")


def ensure_hybrid_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the dense+sparse+ColBERT collection, replacing it when requested."""
    if client.collection_exists(QDRANT_COLLECTION):
        if not recreate:
            return
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIMENSION,
                distance=Distance.COSINE,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            ),
            # ColBERT non indicizzato: solo storage su disco + int8, usato come
            # reranker sui candidati della fusione dense+sparse (hybrid_search).
            "colbert": VectorParams(
                size=DENSE_DIMENSION,
                distance=Distance.COSINE,
                on_disk=True,
                hnsw_config=HnswConfigDiff(m=0),
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=False,
                    )
                ),
            ),
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )


def _point_id(chunk_id: str) -> str:
    """UUID deterministico (name-based) per il chunk_id applicativo."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"esperto:{chunk_id}"))


def upsert_encoded_documents(
    client: QdrantClient,
    documents: Sequence[Document],
    encoded_batch: dict,
) -> None:
    """Upsert BGE-M3-encoded documents, falling back to individual requests if needed."""
    points = [
        PointStruct(
            id=_point_id(document.metadata["chunk_id"]),
            vector={
                "dense": dense_vector.tolist(),
                "sparse": lexical_weights_to_sparse_vector(lexical_weights),
                "colbert": colbert_vector.tolist(),
            },
            payload={"page_content": document.page_content, "metadata": document.metadata},
        )
        for document, dense_vector, lexical_weights, colbert_vector in zip(
            documents,
            encoded_batch["dense_vecs"],
            encoded_batch["lexical_weights"],
            encoded_batch["colbert_vecs"],
        )
    ]
    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    except UnexpectedResponse as error:
        if "larger than allowed" not in str(error) or len(points) == 1:
            raise
        logger.warning("Payload troppo grande; eseguo l'upsert dei %d chunk singolarmente.", len(points))
        for point in points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=[point])


async def _query_colbert(client: AsyncQdrantClient, **kwargs) -> list[Document]:
    """Run a ColBERT query_points call with the shared timeout/error handling."""
    try:
        response = await asyncio.wait_for(
            client.query_points(
                collection_name=QDRANT_COLLECTION,
                using="colbert",
                search_params=_QUANTIZATION_PARAMS,
                **kwargs,
            ),
            timeout=QDRANT_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Qdrant query timeout after %ss", QDRANT_QUERY_TIMEOUT)
        return []
    except Exception as error:
        logger.error("Qdrant query failed: %s: %s", type(error).__name__, error)
        return []

    if not response.points:
        return []

    relevance_threshold = max(point.score for point in response.points) * SEMANTIC_RELEVANCE_RATIO
    return [
        Document(
            page_content=point.payload.get("page_content", ""),
            metadata={
                **point.payload.get("metadata", {}),
                "_colbert_score": point.score,
            },
        )
        for point in response.points
        if point.score >= relevance_threshold
    ]


async def hybrid_search(
    client: AsyncQdrantClient,
    query_vectors: QueryVectors,
    limit: int,
    query_filter: Optional[Filter] = None,
    options: RetrievalOptions = RetrievalOptions(),
) -> list[Document]:
    """Fusione dense+sparse (RRF) seguita da reranking ColBERT sui soli candidati."""
    prefetch = []
    if options.use_dense:
        prefetch.append(Prefetch(query=query_vectors.dense, using="dense", limit=PREFETCH_LIMIT, filter=query_filter))
    if options.use_sparse:
        prefetch.append(Prefetch(query=query_vectors.sparse, using="sparse", limit=PREFETCH_LIMIT, filter=query_filter))
    try:
        response = await asyncio.wait_for(
            client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=RERANK_CANDIDATE_LIMIT if options.use_colbert else limit,
                query_filter=query_filter,
                search_params=_QUANTIZATION_PARAMS,
                with_payload=not options.use_colbert,
            ),
            timeout=QDRANT_QUERY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Qdrant fusion query timeout after %ss", QDRANT_QUERY_TIMEOUT)
        return []
    except Exception as error:
        logger.error("Qdrant fusion query failed: %s: %s", type(error).__name__, error)
        return []

    if not response.points:
        return []

    if not options.use_colbert:
        # Lo score RRF e' salvato nella stessa chiave dello score ColBERT.
        return [
            Document(
                page_content=point.payload.get("page_content", ""),
                metadata={**point.payload.get("metadata", {}), "_colbert_score": point.score},
            )
            for point in response.points
        ]

    # I candidati rispettano gia' query_filter: il rerank li riordina per ID.
    return await _query_colbert(
        client,
        query=query_vectors.colbert,
        limit=limit,
        query_filter=Filter(must=[HasIdCondition(has_id=[point.id for point in response.points])]),
        with_payload=True,
    )


async def rerank_chunk_ids(
    client: AsyncQdrantClient,
    query_vectors: QueryVectors,
    chunk_ids: list[str],
) -> list[Document]:
    """Score already-known chunk IDs with ColBERT only, skipping the dense+sparse prefetch."""
    if not chunk_ids:
        return []
    return await _query_colbert(
        client,
        query=query_vectors.colbert,
        limit=len(chunk_ids),
        query_filter=Filter(
            must=[HasIdCondition(has_id=[_point_id(chunk_id) for chunk_id in chunk_ids])]
        ),
        with_payload=True,
    )