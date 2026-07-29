"""Compatibility facade for the Esperto RAG streaming API."""

from .core.rag.chain import rag_chain, stream_expert_response

__all__ = ["rag_chain", "stream_expert_response"]