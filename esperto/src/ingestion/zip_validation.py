"""Validazione sicura degli archivi ZIP caricati per l'indicizzazione."""

from __future__ import annotations

import io
import os
import zipfile

# Estensioni ammesse: ricavate da env SUPPORTED_EXTENSIONS (stessa lista usata
# dalla pipeline di indicizzazione), con fallback alla lista storica del progetto.
_DEFAULT_EXTENSIONS = ".py,.yaml,.yml,.md,.txt,.html,.json,.js"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
SUPPORTED_EXTENSIONS = frozenset(
    ext.strip()
    for ext in os.getenv("SUPPORTED_EXTENSIONS", _DEFAULT_EXTENSIONS).split(",")
    if ext.strip()
)


class ZipValidationError(ValueError):
    """Errore di validazione dell'archivio ZIP caricato."""


def validate_zip(content: bytes, filename: str) -> list[str]:
    """
    Valida un archivio ZIP caricato e restituisce la lista dei nomi file ammessi.

    Args:
        content: Contenuto binario del file caricato.
        filename: Nome originale del file (per messaggi di errore).

    Returns:
        Lista ordinata dei nomi dei file estrabili.

    Raises:
        ZipValidationError: Se l'archivio non e' valido o contiene voci non ammesse.
    """
    if not filename.lower().endswith(".zip"):
        raise ZipValidationError("Il file deve avere estensione .zip")

    if len(content) == 0:
        raise ZipValidationError("Il file caricato e' vuoto")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise ZipValidationError(
            f"Archivio troppo grande: {len(content) / (1024 * 1024):.1f} MB "
            f"(massimo {MAX_UPLOAD_MB} MB)"
        )

    if not content.startswith(b"PK\x03\x04") and not content.startswith(b"PK\x05\x06"):
        raise ZipValidationError("Il file non e' un archivio ZIP valido")

    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as e:
        raise ZipValidationError(f"Archivio ZIP corrotto: {e}") from e

    with archive:
        if archive.testzip() is not None:
            raise ZipValidationError("Archivio ZIP corrotto: checksum non valido")

        names: list[str] = []
        for info in archive.infolist():
            name = info.filename

            if info.is_dir():
                continue

            # Path traversal e path assoluti
            normalized = name.replace("\\", "/")
            if (
                normalized.startswith("/")
                or normalized.startswith("..")
                or "/../" in normalized
                or (len(normalized) > 1 and normalized[1] == ":")
            ):
                raise ZipValidationError(f"Voce non ammessa nell'archivio: {name}")

            suffix = os.path.splitext(normalized)[1].lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                # I file con estensione non supportata vengono ignorati, non rifiutati:
                # la pipeline di indicizzazione li scarterebbe comunque.
                continue

            names.append(normalized)

    if not names:
        raise ZipValidationError(
            "Nessun file indicizzabile nell'archivio "
            f"(estensioni ammesse: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    return sorted(names)
