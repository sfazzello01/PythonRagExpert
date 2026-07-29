"""LangChain composition and streaming entry point for the Esperto RAG."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_MODEL, OLLAMA_URL, TEMPERATURE

from .prompts import build_prompt
from .retrieval import HybridRetriever


def build_rag_chain():
    """Compose retrieval, prompt construction, Ollama generation, and string streaming."""
    retriever = HybridRetriever()
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    return (
        {"context": retriever.retrieve_context, "question": RunnablePassthrough()}
        | build_prompt()
        | llm
        | StrOutputParser()
    )


rag_chain = build_rag_chain()


async def stream_expert_response(question: str):
    """Yield generated answer chunks for the API streaming response."""
    async for chunk in rag_chain.astream(question):
        yield chunk