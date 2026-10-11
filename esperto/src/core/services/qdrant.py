"""Gestione delle collection Qdrant, caricamento dei chunk e ricerca densa."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence

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
    HasIdCondition,
    PointStruct,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from config import (
    DENSE_DIMENSION,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_QUERY_TIMEOUT,
)

logger = logging.getLogger("EspertoQdrant")

# Con scalar quantization attiva, rescore sui vettori originali per la qualità.
_QUANTIZATION_PARAMS = SearchParams(
    quantization=QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0)
)


def get_sync_client() -> QdrantClient:
    """Crea il client sincrono usato dall'indicizzazione e dal controllo di stato.

    Returns:
        Il client Qdrant sincrono.
    """
    return QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")


def get_async_client() -> AsyncQdrantClient:
    """Crea il client asincrono usato dal retrieval durante le richieste.

    Returns:
        Il client Qdrant asincrono.
    """
    return AsyncQdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")


async def get_documents_by_chunk_ids(client: AsyncQdrantClient, chunk_ids: list[str]) -> list[Document]:
    """Legge i chunk indicizzati a partire dai loro ID applicativi.

    Args:
        client: Client Qdrant asincrono.
        chunk_ids: ID dei chunk da leggere.

    Returns:
        I chunk trovati, nell'ordine degli ID richiesti; quelli assenti dall'indice sono omessi.
    """
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
    return [documents_by_chunk_id[chunk_id] for chunk_id in chunk_ids if chunk_id in documents_by_chunk_id]


def wait_for_qdrant() -> QdrantClient:
    """Attende fino a sessanta secondi che Qdrant risponda.

    Returns:
        Il client sincrono collegato.

    Raises:
        RuntimeError: Se Qdrant non risponde entro il tempo massimo.
    """
    client = get_sync_client()
    for attempt in range(1, 31):
        try:
            client.get_collections()
            logger.info("Qdrant è online e risponde.")
            return client
        except Exception as error:
            logger.warning("Attesa Qdrant (%d/30): %s", attempt, error)
            time.sleep(2)
    raise RuntimeError("Qdrant non disponibile dopo 60 secondi.")


def has_active_collection(client: QdrantClient) -> bool:
    """Indica se l'alias QDRANT_COLLECTION punta a un indice costruito.

    Args:
        client: Client Qdrant sincrono.

    Returns:
        True se l'alias esiste.
    """
    return any(alias.alias_name == QDRANT_COLLECTION for alias in client.get_aliases().aliases)


def activate_collection(client: QdrantClient, collection: str) -> None:
    """Sposta atomicamente l'alias QDRANT_COLLECTION su un nuovo indice ed elimina i precedenti.

    Le query usano sempre l'alias: fino allo scambio continuano a vedere l'indice precedente.

    Args:
        client: Client Qdrant sincrono.
        collection: Nome della collection appena costruita.
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


def create_collection(client: QdrantClient, collection: str) -> None:
    """Crea la collection dei vettori densi BGE-M3, con quantizzazione scalare a 8 bit.

    Args:
        client: Client Qdrant sincrono.
        collection: Nome della collection da creare.
    """
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
        },
    )


def _point_id(chunk_id: str) -> str:
    """Calcola l'ID del punto Qdrant di un chunk.

    Args:
        chunk_id: ID applicativo del chunk.

    Returns:
        Un UUID deterministico, basato sul nome.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"esperto:{chunk_id}"))


def upsert_encoded_documents(
    client: QdrantClient,
    collection: str,
    documents: Sequence[Document],
    dense_vectors: Sequence,
) -> None:
    """Carica i chunk con i loro vettori densi.

    Se il batch supera la dimensione ammessa da Qdrant, i chunk sono caricati uno alla volta.

    Args:
        client: Client Qdrant sincrono.
        collection: Collection di destinazione.
        documents: Chunk da caricare.
        dense_vectors: Vettori densi corrispondenti, nello stesso ordine.

    Raises:
        UnexpectedResponse: Se Qdrant rifiuta il caricamento per un motivo diverso dalla dimensione del batch.
    """
    points = [
        PointStruct(
            id=_point_id(document.metadata["chunk_id"]),
            vector={"dense": dense_vector.tolist()},
            payload={"page_content": document.page_content, "metadata": document.metadata},
        )
        for document, dense_vector in zip(documents, dense_vectors)
    ]
    try:
        client.upsert(collection_name=collection, points=points)
    except UnexpectedResponse as error:
        if "larger than allowed" not in str(error) or len(points) == 1:
            raise
        logger.warning("Payload troppo grande; eseguo l'upsert dei %d chunk singolarmente.", len(points))
        for point in points:
            client.upsert(collection_name=collection, points=[point])


async def dense_search(
    client: AsyncQdrantClient,
    query_vector: list[float],
    limit: int,
    query_filter: Filter | None = None,
    min_score_ratio: float = 0.0,
) -> list[Document]:
    """Cerca i chunk più simili alla domanda per similarità coseno.

    Timeout ed errori di Qdrant si propagano: un contesto vuoto sarebbe indistinguibile da "nessun
    risultato".

    Args:
        client: Client Qdrant asincrono.
        query_vector: Vettore denso della domanda.
        limit: Numero massimo di chunk.
        query_filter: Filtro che restringe la ricerca, ad esempio a un file.
        min_score_ratio: Quota della similarità migliore sotto la quale un chunk viene scartato.

    Returns:
        I chunk in ordine di similarità, esposta nel metadato `_score`.
    """
    response = await asyncio.wait_for(
        client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
            search_params=_QUANTIZATION_PARAMS,
            with_payload=True,
        ),
        timeout=QDRANT_QUERY_TIMEOUT,
    )
    threshold = max((point.score for point in response.points), default=0.0) * min_score_ratio
    return [
        Document(
            page_content=point.payload.get("page_content", ""),
            metadata={**point.payload.get("metadata", {}), "_score": point.score},
        )
        for point in response.points
        if point.score >= threshold
    ]


async def rerank_chunk_ids(
    client: AsyncQdrantClient, query_vector: list[float], chunk_ids: list[str]
) -> list[Document]:
    """Valuta chunk già noti rispetto a una sola domanda, così da dare loro score confrontabili.

    Args:
        client: Client Qdrant asincrono.
        query_vector: Vettore denso della domanda.
        chunk_ids: ID dei chunk da valutare.

    Returns:
        I chunk in ordine di similarità, esposta nel metadato `_score`.
    """
    if not chunk_ids:
        return []
    id_filter = Filter(must=[HasIdCondition(has_id=[_point_id(chunk_id) for chunk_id in chunk_ids])])
    return await dense_search(client, query_vector, limit=len(chunk_ids), query_filter=id_filter)
