# rag.py
from __future__ import annotations

import asyncio
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama  # type: ignore
from langchain_qdrant import QdrantVectorStore  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue  # type: ignore
from langchain_core.documents import Document

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    TEMPERATURE,
    K,
)
from .nomic_conf import NomicOllamaEmbeddings, sanitize_text


def _build_rag_chain():
    # 1. Inizializzazione Client Qdrant e Modello Embeddings
    embeddings = NomicOllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
    
    # Inizializzazione VectorStore (con parametro 'embedding' corretto)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # 2. Espansione Query Ottimizzata (3 query mirate invece di 6)
    def expand_queries(question: str) -> list[str]:
        cleaned = sanitize_text(question).strip()
        if not cleaned:
            return []
        return [
            cleaned,
            f"{cleaned} implementation function code",
            f"{cleaned} architectural structure flow",
        ]

    # 3. Recupero del Contesto Asincrono con Parent Retrieval Controllato
    async def retrieve_context(question: str) -> str:
        queries = expand_queries(question)
        if not queries:
            return ""

        # Ricerca iniziale in parallelo per le query espanse
        tasks = [
            vectorstore.asimilarity_search(query=sanitize_text(q), k=K)
            for q in queries
        ]
        results = await asyncio.gather(*tasks)

        # Rimozione duplicati dai risultati iniziali
        seen_contents = set()
        unique_docs: list[Document] = []
        for query_res in results:
            for doc in query_res:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)

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

            # Eseguiamo il Parent Retrieval locale (k=4 chunk adiacenti invece di 10)
            try:
                same_file_docs = await vectorstore.asimilarity_search(
                    query=sanitize_text(question),
                    k=4,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.filename",
                                match=MatchValue(value=filename)
                            )
                        ]
                    )
                )
                final_docs.extend(same_file_docs)
            except Exception:
                # In caso di errore di filtro o Qdrant, facciamo un fallback sul chunk singolo
                final_docs.append(doc)

        # Pulizia finale dei duplicati generati dalle intersezioni del parent retrieval
        final_seen = set()
        context_blocks = []
        for doc in final_docs:
            if doc.page_content not in final_seen:
                final_seen.add(doc.page_content)
                context_blocks.append(doc.page_content)

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