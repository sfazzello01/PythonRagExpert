"""Compatibility facade for the Esperto RAG streaming API."""

from .core.rag.chain import retrieve_only, stream_expert_response

__all__ = ["retrieve_only", "stream_expert_response"]