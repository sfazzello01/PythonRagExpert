"""Text utilities shared by indexing and retrieval."""


def sanitize_text(text: str) -> str:
    """Remove invalid UTF-8 sequences from text before model processing."""
    return text.encode("utf-8", "ignore").decode("utf-8")