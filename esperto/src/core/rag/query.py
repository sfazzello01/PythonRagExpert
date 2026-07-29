"""Deterministic query-expansion policy for codebase retrieval."""

from ..shared.text import sanitize_text


def expand_queries(question: str) -> list[str]:
    """Return the four complementary retrieval queries for a user question."""
    cleaned = sanitize_text(question).strip()
    if not cleaned:
        return []
    return [
        cleaned,
        f"{cleaned} implementation function code",
        f"{cleaned} architectural structure flow",
        f"{cleaned} usage examples exception handling",
    ]