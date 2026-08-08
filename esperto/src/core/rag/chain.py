"""LangChain composition and streaming entry point for the Esperto RAG."""

from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama  # type: ignore

from config import LLM_MODEL, OLLAMA_URL, TEMPERATURE

from .prompts import build_prompt
from .retrieval import HybridRetriever

_retriever = HybridRetriever()
_llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
_generation_chain = build_prompt() | _llm | StrOutputParser()


def build_rag_chain():
    """Compose retrieval, prompt construction, Ollama generation, and string streaming."""
    return (
        {"context": _retriever.retrieve_context, "question": RunnablePassthrough()}
        | build_prompt()
        | _llm
        | StrOutputParser()
    )


rag_chain = build_rag_chain()


async def stream_expert_response(question: str):
    """Yield sources metadata then generated answer chunks for the API streaming response."""
    context, sources = await _retriever.retrieve_with_sources(question)
    yield f"SOURCES:{json.dumps({'sources': sources}, ensure_ascii=False)}\n"
    async for chunk in _generation_chain.astream({"context": context, "question": question}):
        yield chunk