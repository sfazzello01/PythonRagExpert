# rag.py
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama  # type: ignore
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient  # type: ignore
from qdrant_client.http.models import (  # type: ignore
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    SparseVector,
)
from FlagEmbedding import BGEM3FlagModel  # type: ignore

from config import (
    LLM_MODEL,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    TEMPERATURE,
    K,
    PREFETCH_LIMIT,
    QDRANT_QUERY_TIMEOUT,
    LLM_INFERENCE_TIMEOUT,
    SEMANTIC_RELEVANCE_THRESHOLD,
    EMBEDDING_MAX_LENGTH,
)

logger = logging.getLogger("EspertoRAG")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sanitize_text(text: str) -> str:
    """Rimuove caratteri non UTF-8 da una stringa, prevenendo errori di codifica."""
    return text.encode("utf-8", "ignore").decode("utf-8")


# Stesso modello ibrido usato in indicizzatore.py: un solo forward pass
# produce dense + sparse (lexical_weights) + ColBERT, coerenti con lo spazio
# vettoriale della collection. Caricato una sola volta a livello di modulo.
_bge_m3 = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)


class _QueryVectors:
    __slots__ = ("dense", "sparse", "colbert")

    def __init__(self, dense: List[float], sparse: SparseVector, colbert: List[List[float]]):
        self.dense = dense
        self.sparse = sparse
        self.colbert = colbert


def _lexical_weights_to_sparse_vector(lexical_weights: Dict[str, float]) -> SparseVector:
    if not lexical_weights:
        return SparseVector(indices=[], values=[])
    indices, values = zip(*((int(token_id), float(weight)) for token_id, weight in lexical_weights.items()))
    return SparseVector(indices=list(indices), values=list(values))


def _encode_queries(texts: List[str]) -> List[_QueryVectors]:
    """Encoding SINCRONO (PyTorch) di N query in un solo forward pass.

    Va sempre chiamata tramite asyncio.to_thread: è una chiamata bloccante
    e non deve mai girare direttamente nell'event loop di FastAPI.
    """
    output = _bge_m3.encode(
        texts,
        batch_size=len(texts),
        max_length=EMBEDDING_MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    return [
        _QueryVectors(
            dense=dense_vec.tolist(),
            sparse=_lexical_weights_to_sparse_vector(lexical_weight),
            colbert=colbert_vec.tolist(),
        )
        for dense_vec, lexical_weight, colbert_vec in zip(
            output["dense_vecs"], output["lexical_weights"], output["colbert_vecs"]
        )
    ]


def _build_rag_chain():
    # 1. Inizializzazione Client Qdrant (async, niente più embedding esterno via Ollama)
    client = AsyncQdrantClient(url=f"http://{QDRANT_HOST}:6333")

    # 2. Espansione Query Ottimizzata (4 query mirate: semantica originale, implementazione, architettura, usage)
    def expand_queries(question: str) -> list[str]:
        cleaned = sanitize_text(question).strip()
        if not cleaned:
            return []
        return [
            cleaned,
            f"{cleaned} implementation function code",
            f"{cleaned} architectural structure flow",
            f"{cleaned} usage examples exception handling",
        ]

    async def _hybrid_search(
        qv: _QueryVectors, k: int, query_filter: Optional[Filter] = None
    ) -> List[Document]:
        """Prefetch dense+sparse (candidati veloci) poi rerank finale con ColBERT (MaxSim).
        
        Includes timeout and error handling to prevent stream corruption.
        """
        try:
            response = await asyncio.wait_for(
                client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    prefetch=[
                        Prefetch(query=qv.dense, using="dense", limit=PREFETCH_LIMIT, filter=query_filter),
                        Prefetch(query=qv.sparse, using="sparse", limit=PREFETCH_LIMIT, filter=query_filter),
                    ],
                    query=qv.colbert,
                    using="colbert",
                    limit=k,
                    query_filter=query_filter,
                    with_payload=True,
                ),
                timeout=QDRANT_QUERY_TIMEOUT,
            )
            
            docs = [
                Document(
                    page_content=point.payload.get("page_content", ""),
                    metadata={
                        **point.payload.get("metadata", {}),
                        "_colbert_score": point.score,
                    },
                )
                for point in response.points
                # Filter by semantic relevance threshold if non-zero
                if point.score >= SEMANTIC_RELEVANCE_THRESHOLD
            ]
            logger.debug(f"Hybrid search returned {len(docs)} documents (threshold={SEMANTIC_RELEVANCE_THRESHOLD})")
            return docs
        except asyncio.TimeoutError:
            logger.error(f"Qdrant query timeout after {QDRANT_QUERY_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"Error during hybrid search: {type(e).__name__}: {str(e)}")
            return []

    # 3. Recupero del Contesto Asincrono con Parent Retrieval Controllato
    async def retrieve_context(question: str) -> str:
        queries = expand_queries(question)
        if not queries:
            return ""

        # Un solo forward pass PyTorch per tutte e 4 le query espanse, eseguito
        # in un thread separato per non bloccare l'event loop di FastAPI.
        query_vectors = await asyncio.to_thread(_encode_queries, queries)
        # La prima è sempre la domanda originale: la riusiamo per il parent
        # retrieval invece di ri-encodare (stesso identico embedding).
        original_qv = query_vectors[0]

        # Ricerca iniziale in parallelo per le query espanse
        tasks = [_hybrid_search(qv, k=K) for qv in query_vectors]
        results = await asyncio.gather(*tasks)

        # Per ogni chunk conserviamo il migliore score ottenuto dalle query espanse.
        unique_docs_by_content: dict[str, Document] = {}
        for query_res in results:
            for doc in query_res:
                previous_doc = unique_docs_by_content.get(doc.page_content)
                if previous_doc is None or doc.metadata["_colbert_score"] > previous_doc.metadata["_colbert_score"]:
                    unique_docs_by_content[doc.page_content] = doc

        unique_docs = list(unique_docs_by_content.values())

        final_docs: list[Document] = []
        processed_files = set()

        for doc in unique_docs:
            filename = doc.metadata.get("filename")

            # Mappa globale o documenti privi di file di origine vengono inclusi direttamente
            if not filename or filename == "__codebase_map__":
                final_docs.append(doc)
                continue

            # Se il file è già stato arricchito con il Parent Retrieval, saltiamo per non fare chiamate doppie
            if filename in processed_files:
                continue

            processed_files.add(filename)

            # Eseguiamo il Parent Retrieval locale (k=4 chunk adiacenti invece di 10),
            # riusando gli embedding della domanda originale: nessun nuovo encode.
            try:
                same_file_docs = await _hybrid_search(
                    original_qv,
                    k=4,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.filename",
                                match=MatchValue(value=filename),
                            )
                        ]
                    ),
                )
                final_docs.extend(same_file_docs)
            except Exception as e:
                # In caso di errore di filtro o Qdrant, facciamo un fallback sul chunk singolo
                logger.warning(f"Parent retrieval failed for {filename}: {str(e)}, using single chunk")
                final_docs.append(doc)

        # Anche il parent retrieval puo restituire chunk gia presenti: manteniamo
        # sempre la versione con il punteggio ColBERT piu alto.
        final_docs_by_content: dict[str, Document] = {}
        for doc in final_docs:
            previous_doc = final_docs_by_content.get(doc.page_content)
            if previous_doc is None or doc.metadata["_colbert_score"] > previous_doc.metadata["_colbert_score"]:
                final_docs_by_content[doc.page_content] = doc

        ranked_docs = sorted(
            final_docs_by_content.values(),
            key=lambda doc: doc.metadata["_colbert_score"],
            reverse=True,
        )
        context_blocks = [doc.page_content for doc in ranked_docs]

        return "\n\n---\n\n".join(context_blocks)

    # 4. Strutturazione dei Messaggi di Sistema (System vs Human)
    system_instruction = (
        "Sei un assistente tecnico Senior con accesso diretto alla codebase del progetto.\n"
        "Il tuo compito è analizzare i file forniti nel contesto e rispondere all'utente.\n\n"
        "REGOLE TASSATIVE DI RISPOSTA:\n"
        "1. Usa ESCLUSIVAMENTE il codice e le informazioni presenti nel contesto fornito.\n"
        "2. Se la risposta non è chiaramente deducibile dal codice fornito, di' esplicitamente "
        "\"Non lo so basandomi sul codice fornito\". Non provare ad indovinare o dedurre logiche non visibili.\n"
        "3. Non assumere l'esistenza di librerie, variabili o funzioni di cui non vedi l'implementazione o la firma.\n"
        "4. Se proponi modifiche o refactoring, mostra chiaramente il codice \"Prima\" (quello del contesto) "
        "e il codice \"Dopo\" proposto.\n"
        "5. Cita sempre il nome del file e, se presente, la linea o funzione a cui ti stai riferendo."
    )

    human_instruction = (
        "CONTESTO RECUPERATO DALLA CODEBASE:\n"
        "{context}\n\n"
        "DOMANDA: {question}\n\n"
        "RISPOSTA TECNICA:"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instruction),
        HumanMessagePromptTemplate.from_template(human_instruction)
    ])

    # 5. Configurazione LLM (Qwen 2.5 Coder 3B)
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
    )

    # 6. Catena RAG Finale
    return (
        {"context": retrieve_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Inizializzazione interna della catena RAG
rag_chain = _build_rag_chain()


# ==============================================================================
# FUNZIONE ESPORTATA PER L'API (api.py)
# ==============================================================================
async def stream_expert_response(question: str):
    """
    Consuma in modo asincrono la catena RAG e genera token uno alla volta.
    Questo è il punto di ingresso richiesto da api.py per lo streaming in tempo reale.
    """
    async for chunk in rag_chain.astream(question):
        yield chunk