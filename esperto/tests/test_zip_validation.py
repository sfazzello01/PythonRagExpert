"""Unit test per la validazione sicura degli archivi ZIP di ingestion."""

import io
import zipfile

import pytest

from src.ingestion.zip_validation import ZipValidationError, validate_zip


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def test_valid_zip_returns_only_supported_names():
    content = _zip_bytes(
        {
            "main.py": "print('hello')",
            "docs/readme.md": "# Titolo",
            "notes.bin": "dati binari ignorati",
        }
    )
    assert validate_zip(content, "code.zip") == ["docs/readme.md", "main.py"]


def test_rejects_non_zip_extension():
    with pytest.raises(ZipValidationError, match="estensione .zip"):
        validate_zip(_zip_bytes({"main.py": "x"}), "code.rar")


def test_rejects_empty_upload():
    with pytest.raises(ZipValidationError, match="vuoto"):
        validate_zip(b"", "code.zip")


def test_rejects_non_zip_content():
    with pytest.raises(ZipValidationError, match="non e' un archivio ZIP"):
        validate_zip(b"questo non e' uno zip", "code.zip")


def test_rejects_path_traversal():
    with pytest.raises(ZipValidationError, match="non ammessa"):
        validate_zip(_zip_bytes({"../evil.py": "x"}), "code.zip")


def test_rejects_absolute_path():
    with pytest.raises(ZipValidationError, match="non ammessa"):
        validate_zip(_zip_bytes({"/etc/evil.py": "x"}), "code.zip")


def test_rejects_zip_without_supported_files():
    with pytest.raises(ZipValidationError, match="Nessun file indicizzabile"):
        validate_zip(_zip_bytes({"notes.bin": "x"}), "code.zip")
