"""Validazione sicura degli archivi ZIP caricati per l'indicizzazione."""

from __future__ import annotations

import io
import os
import zipfile

from config import MAX_EXTRACTED_MB, MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS


class ZipValidationError(ValueError):
    """Errore di validazione dell'archivio ZIP caricato."""


def validate_zip(content: bytes, filename: str) -> list[str]:
    """Valida un archivio ZIP caricato e ne elenca i file ammessi.

    Args:
        content: Contenuto binario del file caricato.
        filename: Nome originale del file, di cui si controlla l'estensione.

    Returns:
        I nomi dei file indicizzabili, in ordine alfabetico.

    Raises:
        ZipValidationError: Se l'archivio non è valido, è troppo grande o contiene voci non ammesse.
    """
    if not filename.lower().endswith(".zip"):
        raise ZipValidationError("Il file deve avere estensione .zip")

    if not content:
        raise ZipValidationError("Il file caricato è vuoto")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise ZipValidationError(
            f"Archivio troppo grande: {len(content) / (1024 * 1024):.1f} MB "
            f"(massimo {MAX_UPLOAD_MB} MB)"
        )

    if not content.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise ZipValidationError("Il file non è un archivio ZIP valido")

    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ZipValidationError(f"Archivio ZIP corrotto: {error}") from error

    with archive:
        # Prima di testzip(), che decomprime l'intero archivio (zip bomb).
        if sum(info.file_size for info in archive.infolist()) > MAX_EXTRACTED_MB * 1024 * 1024:
            raise ZipValidationError(f"Archivio troppo grande una volta estratto (massimo {MAX_EXTRACTED_MB} MB)")
        if archive.testzip() is not None:
            raise ZipValidationError("Archivio ZIP corrotto: checksum non valido")

        names: list[str] = []
        for info in archive.infolist():
            name = info.filename

            if info.is_dir():
                continue

            # Path traversal e path assoluti.
            normalized = name.replace("\\", "/")
            if (
                normalized.startswith(("/", ".."))
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
