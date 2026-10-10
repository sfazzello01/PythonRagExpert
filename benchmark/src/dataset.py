"""Caricamento e validazione del dataset: repository a commit fissato e gold set."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import zipfile
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from config import DATASETS_DIR, REPO_CACHE_DIR, SUPPORTED_EXTENSIONS

CATEGORIES = ("lookup", "behavior", "multihop", "architecture", "overview", "unanswerable")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class Relevant(BaseModel):
    """Simbolo che il retrieval deve trovare per rispondere a una domanda."""

    file: str
    symbol: str  # simbolo top-level: per un metodo, la classe che lo contiene
    hop: int = 1  # 2 = raggiungibile solo seguendo una relazione di chiamata/ereditarietà


class Question(BaseModel):
    """Domanda del gold set, con i simboli attesi e i fatti chiave della risposta."""

    id: str
    category: str
    question: str
    relevant: list[Relevant] = []
    key_facts: list[str] = []
    reference_answer: str = ""

    @property
    def answerable(self) -> bool:
        """True se la domanda ha una risposta nel codice."""
        return self.category != "unanswerable"

    @model_validator(mode="after")
    def _check_shape(self) -> Question:
        """Verifica che la domanda abbia il gold previsto dalla sua categoria.

        Returns:
            La domanda stessa, se è valida.

        Raises:
            ValueError: Se la categoria è sconosciuta o il gold non è coerente con la categoria.
        """
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: categoria sconosciuta {self.category!r}")
        if self.answerable and not self.key_facts:
            raise ValueError(f"{self.id}: servono i key_facts")
        # Le domande di panoramica non hanno un simbolo da recuperare: si valutano solo sulla risposta.
        if self.answerable and self.category != "overview" and not self.relevant:
            raise ValueError(f"{self.id}: servono i simboli rilevanti")
        if not self.answerable and (self.relevant or self.key_facts):
            raise ValueError(f"{self.id}: una domanda senza risposta non ha gold")
        return self


class Dataset(BaseModel):
    """Dataset di benchmark: repository, riferimento Git e domande del gold set."""

    name: str
    repo_url: str
    ref: str
    questions: list[Question]


def check_name(name: str) -> str:
    """Verifica che un nome di dataset sia sicuro da usare come nome di cartella.

    Args:
        name: Nome da verificare.

    Returns:
        Il nome, se è valido.

    Raises:
        ValueError: Se il nome contiene caratteri diversi da lettere, cifre, `_` e `-`.
    """
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Nome non valido: {name!r}")
    return name


def load_dataset(name: str) -> Dataset:
    """Carica un dataset dalla cartella dei dataset.

    Args:
        name: Nome del dataset.

    Returns:
        Il dataset con le sue domande.

    Raises:
        ValueError: Se il nome non è valido, una domanda non rispetta lo schema o ci sono ID duplicati.
    """
    folder = DATASETS_DIR / check_name(name)
    meta = yaml.safe_load((folder / "dataset.yaml").read_text(encoding="utf-8"))
    questions = yaml.safe_load((folder / "questions.yaml").read_text(encoding="utf-8"))
    dataset = Dataset(name=name, questions=questions, **meta)
    ids = [question.id for question in dataset.questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Id domanda duplicati nel dataset")
    return dataset


def fetch_repo(dataset: Dataset) -> tuple[Path, str]:
    """Clona il repository al riferimento indicato, riusando la copia in cache se esiste.

    Args:
        dataset: Dataset di cui clonare il repository.

    Returns:
        La cartella del checkout e l'hash del suo commit.
    """
    destination = REPO_CACHE_DIR / f"{dataset.name}-{dataset.ref}"
    if not destination.exists():
        REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", dataset.ref, dataset.repo_url, str(destination)],
            check=True,
            capture_output=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return destination, commit


def list_repo_files(repo_dir: Path) -> list[str]:
    """Elenca i file indicizzabili del repository.

    Args:
        repo_dir: Cartella del checkout.

    Returns:
        I percorsi relativi alla radice, come li vede esperto, in ordine alfabetico.
    """
    return sorted(
        path.relative_to(repo_dir).as_posix()
        for path in repo_dir.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(repo_dir).parts
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_zip(repo_dir: Path, files: list[str]) -> bytes:
    """Crea lo ZIP della codebase da caricare su esperto.

    I file stanno alla radice dell'archivio, così i `filename` indicizzati coincidono con il gold.

    Args:
        repo_dir: Cartella del checkout.
        files: Percorsi relativi dei file da includere.

    Returns:
        I byte dell'archivio.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(repo_dir / name, arcname=name)
    return buffer.getvalue()


def _top_level_symbols(path: Path) -> set[str] | None:
    """Elenca funzioni e classi di primo livello di un file Python.

    Args:
        path: File da analizzare.

    Returns:
        I nomi dei simboli, oppure None se il file ha un errore di sintassi.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_dataset(dataset: Dataset, repo_dir: Path) -> list[str]:
    """Confronta il gold set con il checkout del repository.

    Args:
        dataset: Dataset da verificare.
        repo_dir: Cartella del checkout.

    Returns:
        Un messaggio per ogni file o simbolo del gold che non esiste nel checkout.
    """
    errors: list[str] = []
    for question in dataset.questions:
        for item in question.relevant:
            path = repo_dir / item.file
            if not path.is_file():
                errors.append(f"{question.id}: file inesistente {item.file}")
                continue
            symbols = _top_level_symbols(path)
            if symbols is None or item.symbol not in symbols:
                errors.append(f"{question.id}: simbolo top-level {item.symbol!r} non trovato in {item.file}")
    return errors
