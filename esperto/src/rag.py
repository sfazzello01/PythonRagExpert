"""Compatibility facade for the Esperto RAG streaming API."""

from .core.rag.chain import stream_expert_response

__all__ = ["stream_expert_response"]