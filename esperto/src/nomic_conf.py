from typing import Any
from langchain_ollama import OllamaEmbeddings # type: ignore

def sanitize_text(text: str) -> str:
    """
    Rimuove caratteri non UTF-8 da una stringa, prevenendo errori di codifica.

    Args:
        text: La stringa da sanificare.

    Returns:
        La stringa sanificata, con solo caratteri UTF-8 validi.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


class FilteredOllamaEmbeddings(OllamaEmbeddings):
    @property
    def _default_params(self) -> dict[str, Any]:
        """
        Restituisce i parametri di default per le embedding, filtrando quelli con valore None.

        Returns:
            Un dizionario di parametri con solo valori non None.
        """
        return {
            key: value
            for key, value in super()._default_params.items()
            if value is not None
        }


class NomicOllamaEmbeddings(FilteredOllamaEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embedding per una lista di testi, prefissandoli con un prefisso specifico.

        Args:
            texts: Una lista di stringhe per cui generare embedding.

        Returns:
            Una lista di embedding per i testi forniti.
        """
        prefixed = [f"search_document: {sanitize_text(t)}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        """
        Genera l'embedding per una singola query, prefissandola con un prefisso specifico.

        Args:
            text: La stringa per cui generare l'embedding.

        Returns:
            L'embedding della query fornita.
        """
        return super().embed_query(f"search_query: {sanitize_text(text)}")
