"""Stato persistente che permette di saltare la ricostruzione di un indice invariato."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import (
    BASE_DIR,
    CHUNKING_VERSION,
    DENSE_DIMENSION,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_ID,
    INDEX_COMPLETE_FILE,
    INDEX_STATE_DIR,
    INDEX_STATE_FILE,
    QDRANT_COLLECTION,
    REPO_MAP_FILE,
    SYMBOL_GRAPH_FILE,
    SYMBOL_GRAPH_VERSION,
)


def source_fingerprint(files: list[Path]) -> str:
    """Calcola l'impronta dei file indicizzati e delle impostazioni che determinano l'indice.

    Args:
        files: File da indicizzare, nell'ordine in cui vengono letti.

    Returns:
        L'hash SHA-256 esadecimale di modello, dimensione, lunghezza massima, versione del chunking e
        contenuto dei file.
    """
    digest = hashlib.sha256()
    digest.update(
        f"{EMBEDDING_MODEL_ID}:{DENSE_DIMENSION}:{EMBEDDING_MAX_LENGTH}:{CHUNKING_VERSION}".encode("utf-8")
    )
    for path in files:
        try:
            relative_path = path.relative_to(BASE_DIR).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def is_current(fingerprint: str) -> bool:
    """Indica se lo stato salvato corrisponde ai sorgenti e alla configurazione attuali.

    Args:
        fingerprint: Impronta dei sorgenti correnti.

    Returns:
        True se collection, impronta e versione del grafo coincidono e i file del grafo e della mappa
        esistono.
    """
    try:
        state = json.loads(INDEX_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        state.get("collection") == QDRANT_COLLECTION
        and state.get("fingerprint") == fingerprint
        and state.get("symbol_graph_version") == SYMBOL_GRAPH_VERSION
        and SYMBOL_GRAPH_FILE.exists()
        and REPO_MAP_FILE.exists()
    )


def save(fingerprint: str, source_file_count: int) -> None:
    """Salva lo stato di un'indicizzazione riuscita e ne crea il file di completamento.

    Args:
        fingerprint: Impronta dei sorgenti indicizzati.
        source_file_count: Numero di file indicizzati.
    """
    INDEX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "collection": QDRANT_COLLECTION,
        "fingerprint": fingerprint,
        "source_file_count": source_file_count,
        "symbol_graph_version": SYMBOL_GRAPH_VERSION,
    }
    temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_file.replace(INDEX_STATE_FILE)
    INDEX_COMPLETE_FILE.touch()
