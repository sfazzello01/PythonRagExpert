"""Ricerca densa, arricchimento con stesso file e symbol graph, rerank finale e budget del contesto."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from itertools import chain

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient  # type: ignore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore

from config import (
    RAG_CONTEXT_MAX_CHARS,
    RAG_CONTEXT_MAX_SOURCES,
    SEMANTIC_RELEVANCE_RATIO,
    SYMBOL_GRAPH_MAX_NEIGHBORS,
    K,
)

from ..services.embedding import encode_queries
from ..services.qdrant import (
    dense_search,
    get_async_client,
    get_documents_by_chunk_ids,
    rerank_chunk_ids,
)
from ..services.symbol_graph import SymbolGraph
from ..shared.options import RetrievalOptions
from ..shared.text import sanitize_text, without_imports

# Chunk più simili alla domanda presi, al massimo, da ogni file dei risultati della ricerca.
SAME_FILE_LIMIT = 4
# Risultati i cui vicini diretti nel grafo hanno la precedenza sugli altri.
TOP_SEEDS = 3
# Stadio che porta un chunk tra i candidati, dal più semplice: se più stadi trovano lo stesso
# chunk vale il primo, così si legge quanto ogni stadio aggiunge alla sola ricerca.
ORIGINS = ("search", "symbol", "same_file", "graph")


def _keep_highest_scoring(documents: Iterable[Document]) -> list[Document]:
    """Elimina i chunk duplicati per chunk_id, conservando per ciascuno lo score migliore.

    Args:
        documents: Chunk con lo score nel metadato `_score`.

    Returns:
        Un documento per chunk_id, nell'ordine della prima comparsa.
    """
    by_id: dict[str, Document] = {}
    for document in documents:
        previous = by_id.get(document.metadata["chunk_id"])
        if previous is None or document.metadata["_score"] > previous.metadata["_score"]:
            by_id[document.metadata["chunk_id"]] = document
    return list(by_id.values())


def _apply_context_budget(documents: Iterable[Document], max_sources: int, max_chars: int) -> list[Document]:
    """Seleziona i documenti che rientrano nei limiti di numero e di caratteri del contesto.

    Gli import del modulo contano una volta per file, come nel contesto mostrato al modello. Un
    documento che farebbe superare i caratteri viene saltato (il primo entra comunque) e si prova con i
    successivi.

    Args:
        documents: Documenti in ordine di priorità.
        max_sources: Numero massimo di documenti.
        max_chars: Numero massimo di caratteri complessivi.

    Returns:
        I documenti selezionati, nell'ordine ricevuto.
    """
    selected: list[Document] = []
    filenames: set[str] = set()
    total_characters = 0
    for document in documents:
        if len(selected) >= max_sources:
            break
        filename = document.metadata.get("filename")
        content = document.page_content
        size = len(without_imports(content) if filename in filenames else content)
        if selected and total_characters + size > max_chars:
            continue
        selected.append(document)
        filenames.add(filename)
        total_characters += size
    return selected


def _to_source(document: Document, score: float | None, origin: str) -> dict:
    """Converte un chunk nella fonte restituita dall'API.

    Args:
        document: Chunk recuperato.
        score: Similarità con la domanda originale; None per i vicini del grafo.
        origin: Stadio che ha portato il chunk tra i candidati.

    Returns:
        I metadati del chunk con score, origine, relazione nel grafo e testo.
    """
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
        "origin": origin,
        "graph_relation": metadata.get("_graph_relation"),
        "content": document.page_content,
    }


class Retriever:
    """Recupera dall'indice Qdrant (BGE-M3) il contesto ordinato con cui rispondere a una domanda."""

    def __init__(self, client: AsyncQdrantClient | None = None):
        """Crea il retriever.

        Args:
            client: Client Qdrant asincrono; se manca ne viene creato uno.
        """
        self._client = client or get_async_client()
        self._symbol_graph = SymbolGraph()

    async def _candidates(
        self, question: str, query_vector: list[float], options: RetrievalOptions
    ) -> list[Document]:
        """Raccoglie i chunk candidati per una domanda.

        I candidati vengono da ricerca densa, simboli nominati nella domanda, altri chunk degli stessi file
        e vicini nel symbol graph; ognuno porta nel metadato `_origin` lo stadio che l'ha trovato.

        Args:
            question: Domanda o sotto-domanda.
            query_vector: Vettore denso della domanda.
            options: Interruttori degli stadi di retrieval.

        Returns:
            I candidati: prima i simboli nominati, poi i risultati della ricerca con i chunk degli stessi
            file, infine i vicini del grafo.
        """
        named_ids = self._symbol_graph.chunks_named(question) if options.use_symbol_lookup else []
        # La soglia di rilevanza vale solo qui: chunk dello stesso file e rerank finale
        # servono a dare contesto e a ordinare, non a selezionare.
        documents, named = await asyncio.gather(
            dense_search(
                self._client, query_vector, limit=options.k or K, min_score_ratio=SEMANTIC_RELEVANCE_RATIO
            ),
            get_documents_by_chunk_ids(self._client, named_ids),
        )
        for document in named:
            document.metadata["_origin"] = "symbol"
        searched = {document.metadata["chunk_id"] for document in documents}
        # Il grafo parte dai soli semi: risultati della ricerca e simboli nominati nella domanda.
        seeds = [document.metadata["chunk_id"] for document in chain(named, documents)]
        if options.use_same_file:
            filenames = dict.fromkeys(document.metadata["filename"] for document in documents)
            same_file = await asyncio.gather(
                *(
                    dense_search(
                        self._client,
                        query_vector,
                        limit=SAME_FILE_LIMIT,
                        query_filter=Filter(
                            must=[FieldCondition(key="metadata.filename", match=MatchValue(value=filename))]
                        ),
                    )
                    for filename in filenames
                )
            )
            documents = _keep_highest_scoring(chain(documents, *same_file))
        for document in documents:
            document.metadata["_origin"] = "search" if document.metadata["chunk_id"] in searched else "same_file"
        documents = [*named, *documents]
        if not options.use_symbol_graph:
            return documents

        relations = self._symbol_graph.related_chunks(
            set(seeds), limit=SYMBOL_GRAPH_MAX_NEIGHBORS, top_chunk_ids=seeds[:TOP_SEEDS]
        )
        graph_documents = await get_documents_by_chunk_ids(self._client, list(relations))
        for document in graph_documents:
            document.metadata["_origin"] = "graph"
            document.metadata["_graph_relation"] = relations[document.metadata["chunk_id"]]
        return [*documents, *graph_documents]

    async def retrieve(
        self,
        questions: list[str],
        original_question: str,
        options: RetrievalOptions = RetrievalOptions(),
    ) -> tuple[list[dict], list[dict]]:
        """Recupera le fonti per la domanda originale e per le sotto-domande del planner.

        Tutti i candidati sono rivalutati sulla domanda originale e poi tagliati dal budget del contesto.

        Args:
            questions: Sotto-domande del planner.
            original_question: Domanda dell'utente.
            options: Interruttori degli stadi di retrieval.

        Returns:
            Le fonti in ordine di rilevanza e i candidati rimasti fuori dal contesto (senza testo): al
            benchmark servono per distinguere un chunk mai trovato da uno trovato e tagliato dal budget.
        """
        original_question = sanitize_text(original_question).strip()
        retrieval_questions = [
            original_question,
            *(question for question in questions if question.casefold() != original_question.casefold()),
        ]
        # Un solo passaggio BGE-M3 per tutte le domande; il primo vettore è la domanda originale.
        query_vectors = await asyncio.to_thread(encode_queries, retrieval_questions)
        candidates = await asyncio.gather(
            *(
                self._candidates(question, vectors, options)
                for question, vectors in zip(retrieval_questions, query_vectors)
            )
        )
        documents: dict[str, Document] = {}
        neighbors: dict[str, Document] = {}
        named: set[str] = set()
        origins: dict[str, str] = {}
        for document in chain.from_iterable(candidates):
            chunk_id, origin = document.metadata["chunk_id"], document.metadata["_origin"]
            (neighbors if origin == "graph" else documents).setdefault(chunk_id, document)
            if origin == "symbol":
                named.add(chunk_id)
            origins[chunk_id] = min(origin, origins.get(chunk_id, origin), key=ORIGINS.index)

        # Gli score di ricerche diverse (sotto-domande, stesso file) non sono confrontabili:
        # tutti i candidati vengono rivalutati sulla domanda originale.
        reranked = await rerank_chunk_ids(self._client, query_vectors[0], list(documents))
        scores = {document.metadata["chunk_id"]: document.metadata["_score"] for document in reranked}
        # I simboli nominati nella domanda stanno in testa, anche se il rerank non li ha valutati.
        ranked = sorted(
            (document for chunk_id, document in documents.items() if chunk_id in scores or chunk_id in named),
            key=lambda document: (
                document.metadata["chunk_id"] not in named,
                -scores.get(document.metadata["chunk_id"], 0.0),
            ),
        )
        # I vicini del grafo servono proprio quando non somigliano alla domanda: saltano il
        # rerank e hanno una quota riservata per ogni domanda, proporzionale ai loro posti
        # e mai oltre metà del budget.
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
        sources = []
        for document in chain(ranked_documents, graph_documents):  # i vicini del grafo non hanno score
            chunk_id = document.metadata["chunk_id"]
            sources.append(_to_source(document, scores.get(chunk_id), origins[chunk_id]))
        kept = {source["chunk_id"] for source in sources}
        dropped = [
            {
                "filename": document.metadata.get("filename"),
                "symbol_name": document.metadata.get("symbol_name"),
                "chunk_id": chunk_id,
                "origin": origins[chunk_id],
            }
            for chunk_id, document in {**neighbors, **documents}.items()
            if chunk_id not in kept
        ]
        return sources, dropped
