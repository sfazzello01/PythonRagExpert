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
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
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

from config import (
    DENSE_DIMENSION,
    PREFETCH_LIMIT,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_QUERY_TIMEOUT,
)
from ..shared.options import RetrievalOptions
from ..shared.vectors import QueryVectors, lexical_weights_to_sparse_vector

logger = logging.getLogger("EspertoQdrant")

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
    records = await client.retrieve(
        collection_name=QDRANT_COLLECTION,
        ids=[_point_id(chunk_id) for chunk_id in chunk_ids],
        with_payload=True,
    )
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


def has_active_collection(client: QdrantClient) -> bool:
    """Return whether the QDRANT_COLLECTION alias points to a built index."""
    return any(alias.alias_name == QDRANT_COLLECTION for alias in client.get_aliases().aliases)


def activate_collection(client: QdrantClient, collection: str) -> None:
    """Atomically point the QDRANT_COLLECTION alias to a new index and drop the previous ones.

    Le query usano sempre l'alias: fino allo scambio continuano a vedere l'indice precedente.
    """
    names = [description.name for description in client.get_collections().collections]
    if QDRANT_COLLECTION in names:  # indice creato prima degli alias: il nome va liberato
        client.delete_collection(QDRANT_COLLECTION)
    operations = [
        CreateAliasOperation(
            create_alias=CreateAlias(collection_name=collection, alias_name=QDRANT_COLLECTION)
        )
    ]
    if has_active_collection(client):
        operations.insert(0, DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=QDRANT_COLLECTION)))
    client.update_collection_aliases(change_aliases_operations=operations)
    for name in names:
        if name.startswith(f"{QDRANT_COLLECTION}_") and name != collection:
            client.delete_collection(name)


def create_hybrid_collection(client: QdrantClient, collection: str) -> None:
    """Create a dense+sparse+ColBERT collection."""
    client.create_collection(
        collection_name=collection,
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
    collection: str,
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
        client.upsert(collection_name=collection, points=points)
    except UnexpectedResponse as error:
        if "larger than allowed" not in str(error) or len(points) == 1:
            raise
        logger.warning("Payload troppo grande; eseguo l'upsert dei %d chunk singolarmente.", len(points))
        for point in points:
            client.upsert(collection_name=collection, points=[point])


async def _query_colbert(client: AsyncQdrantClient, **kwargs) -> list[Document]:
    """Run a ColBERT query_points call with the shared timeout.

    Timeout ed errori di Qdrant si propagano: un contesto vuoto sarebbe
    indistinguibile da "nessun risultato".
    """
    response = await asyncio.wait_for(
        client.query_points(
            collection_name=QDRANT_COLLECTION,
            using="colbert",
            search_params=_QUANTIZATION_PARAMS,
            **kwargs,
        ),
        timeout=QDRANT_QUERY_TIMEOUT,
    )

    if not response.points:
        return []

    return [
        Document(
            page_content=point.payload.get("page_content", ""),
            metadata={
                **point.payload.get("metadata", {}),
                "_colbert_score": point.score,
            },
        )
        for point in response.points
    ]


async def hybrid_search(
    client: AsyncQdrantClient,
    query_vectors: QueryVectors,
    limit: int,
    query_filter: Optional[Filter] = None,
    options: RetrievalOptions = RetrievalOptions(),
    min_score_ratio: float = 0.0,
) -> list[Document]:
    """Fusione dense+sparse (RRF) seguita da reranking ColBERT sui soli candidati.

    Con `min_score_ratio` il rerank scarta i chunk sotto quella quota dello score migliore.
    """
    prefetch = []
    if options.use_dense:
        prefetch.append(Prefetch(query=query_vectors.dense, using="dense", limit=PREFETCH_LIMIT, filter=query_filter))
    if options.use_sparse:
        prefetch.append(Prefetch(query=query_vectors.sparse, using="sparse", limit=PREFETCH_LIMIT, filter=query_filter))
    response = await asyncio.wait_for(
        client.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            # Con ColBERT il rerank riceve tutti i candidati dei prefetch.
            limit=len(prefetch) * PREFETCH_LIMIT if options.use_colbert else limit,
            query_filter=query_filter,
            search_params=_QUANTIZATION_PARAMS,
            with_payload=not options.use_colbert,
        ),
        timeout=QDRANT_QUERY_TIMEOUT,
    )

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
    documents = await _query_colbert(
        client,
        query=query_vectors.colbert,
        limit=limit,
        query_filter=Filter(must=[HasIdCondition(has_id=[point.id for point in response.points])]),
        with_payload=True,
    )
    best_score = max((document.metadata["_colbert_score"] for document in documents), default=0.0)
    threshold = best_score * min_score_ratio
    return [document for document in documents if document.metadata["_colbert_score"] >= threshold]


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