from langchain_qdrant import QdrantVectorStore  # type: ignore
from langchain_ollama import ChatOllama  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_core.runnables import RunnablePassthrough  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from typing import AsyncIterator
import asyncio

from .nomic_conf import NomicOllamaEmbeddings, sanitize_text
from config import (
    EMBEDDING_MODEL,
    K,
    LLM_MODEL,
    OLLAMA_URL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    TEMPERATURE,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

_chain = None  # global variable per la catena RAG, inizializzata on demand
_retriever = None  # global variable per il retriever, inizializzata on demand


def prompt_str() -> str:
    template = """Sei un assistente tecnico Senior esperto della codebase.
Usa i seguenti frammenti di codice recuperati per rispondere alla domanda.
Se non conosci la risposta basandoti solo sul codice fornito, di' che non lo sai, non inventare.
Le risposte devono essere tecniche, precise e basate esclusivamente sul codice fornito. Non fare supposizioni o aggiunte non supportate dal codice.

Regole:
1. Identifica prima quali file sono coinvolti nella richiesta.
2. Rispondi in modo corretto e completo, ma solo se il codice fornito supporta la risposta.
3. Se il codice è insufficiente per rispondere, ammettilo chiaramente ("Non lo so" o simile).
4. Se proponi modifiche, mostra chiaramente il "Prima" e il "Dopo".
5. Mostra sempre le fonti del tuo ragionamento, indicando file, funzioni, classi, endpoint o template specifici quando sono presenti nel contesto.
6. Quando la domanda chiede "come funziona" qualcosa, spiega il flusso operativo completo usando solo i passaggi supportati dai frammenti recuperati.
7. Usa la mappa della codebase recuperata per orientarti tra moduli, funzioni, classi, route e template prima di concludere che il contesto è insufficiente.

CONTESTO RECUPERATO:
{context}

DOMANDA: {question}

RISPOSTA ESPERTA:"""
    return template


def _build_retriever() -> BaseRetriever:
    """
    Costruisce il retriever per la ricerca di documenti.

    return:
        BaseRetriever: Un retriever configurato per cercare nei documenti indicizzati.
    """
    # 1. Connessione a Qdrant e setup del Retriever
    embeddings = NomicOllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=f"http://{QDRANT_HOST}:6333",
    )

    # 2. Configurazione del Retriever (cerca i k pezzi di codice più rilevanti)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K,
            "fetch_k": max(K * 4, 30),
            "lambda_mult": 0.5,
        },
    )


def _build_rag_chain(
    retriever,
) -> Runnable:
    """
    Costruisce la catena RAG (Retrieval-Augmented Generation).

    args:
        retriever (BaseRetriever): Il retriever per la ricerca di documenti.

    returns:
        Runnable: La catena RAG configurata.
    """
    # 3. Definizione del System Prompt
    # Qui istruiamo il modello a comportarsi da esperto

    prompt = ChatPromptTemplate.from_template(prompt_str())

    # 4. Inizializzazione del Modello
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)

    # 5. La Catena RAG (LCEL - LangChain Expression Language)
    def expand_queries(question: str) -> list[str]:
        cleaned = sanitize_text(question).strip()
        if not cleaned:
            return []

        return [
            cleaned,
            f"{cleaned} implementazione codice funzione classe modulo file",
            f"{cleaned} flusso architettura dipendenze input output configurazione",
        ]

    def retrieve_docs(question: str):
        docs = []
        seen: set[tuple[str, str]] = set()

        for query in expand_queries(question):
            for doc in retriever.invoke(sanitize_text(query)):
                key = (
                    str(doc.metadata.get("filename", "")),
                    doc.page_content[:300],
                )
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)

        return docs[: max(K * 2, K)]

    def format_docs(docs):
        # Uniamo il contenuto dei file e aggiungiamo i metadati per aiutare il modello
        formatted = []
        for doc in docs:
            header = f"--- File: {doc.metadata.get('filename')} (Servizio: {doc.metadata.get('service')}) ---"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def retrieve_context(question: str) -> str:
        return format_docs(retrieve_docs(question))

    return (
        {"context": retrieve_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def _get_retriever() -> BaseRetriever:
    """
    Restituisce il retriever per la ricerca di documenti.

    returns:
        BaseRetriever: Un retriever configurato per cercare nei documenti indicizzati.
    """
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


def _get_chain() -> (
    Runnable
):
    """
    Restituisce la catena RAG (Retrieval-Augmented Generation) per generare risposte basate sui documenti recuperati.

    returns:
        Runnable: La catena RAG configurata.
    """
    global _chain
    if _chain is None:
        _chain = _build_rag_chain(_get_retriever())
    return _chain


async def stream_expert_response(user_query: str) -> AsyncIterator[str]:
    """
    Restituisce la risposta dell'esperto in streaming, chunk per chunk.

    args:
        user_query (str): La domanda posta dall'utente.

    yields:
        str: La risposta generata dall'esperto, restituita in streaming.
    """
    try:
        async for chunk in _get_chain().astream(sanitize_text(user_query)):
            if chunk is not None:
                yield str(chunk)
    except Exception as e:
        yield (
            "Errore durante la generazione della risposta dell'esperto: "
            f"{type(e).__name__}: {e}"
        )


async def get_expert_response(user_query: str) -> str:
    """
    Restituisce la risposta completa dell'esperto.

    args:
        user_query (str): La domanda posta dall'utente.

    returns:
        str: La risposta completa generata dall'esperto come stream.
    """

    full_response = []
    async for chunk in stream_expert_response(user_query):
        print(chunk, end="", flush=True)
        full_response.append(chunk)
    print()

    return "".join(full_response)


def test_retriever() -> None:
    """
    Funzione di test per verificare che il retriever funzioni correttamente.
    """
    retriever = _get_retriever()
    query = input("Inserisci una domanda per testare il retriever: ")
    docs = retriever.invoke(sanitize_text(query))
    for doc in docs:
        print(f"--- Documento recuperato: {doc.metadata.get('filename')} ---")
        print(doc.page_content[:500])  # Stampa i primi 500 caratteri del documento
        print("\n\n")


async def main() -> None:
    """main, loop principale per interagire con l'esperto tramite terminale."""
    while 1:
        scelta = input(
            "premi (1) per testare il retriever, premi (2) per fare una domanda all'esperto: "
        )
        if scelta == "1":
            test_retriever()
        elif scelta == "2":
            domanda = input("Fai una domanda alla codebase (o 'exit' per uscire): ")
            if domanda.lower() == "exit":
                break
            print("L'esperto sta pensando...")
            await get_expert_response(domanda)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
