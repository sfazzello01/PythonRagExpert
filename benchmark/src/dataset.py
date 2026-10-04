"""Caricamento e validazione del dataset (repo a commit fissato + gold set)."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, model_validator

from config import DATASETS_DIR, REPO_CACHE_DIR, SUPPORTED_EXTENSIONS

CATEGORIES = ("lookup", "behavior", "multihop", "architecture", "unanswerable")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class Relevant(BaseModel):
    file: str
    symbol: str  # simbolo top-level: per un metodo, la classe che lo contiene
    hop: int = 1  # 2 = raggiungibile solo seguendo una relazione di chiamata/ereditarieta'

    @property
    def symbol_key(self) -> str:
        return f"{self.file}::{self.symbol}"


class Question(BaseModel):
    id: str
    category: str
    question: str
    relevant: List[Relevant] = []
    key_facts: List[str] = []
    reference_answer: str = ""

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @model_validator(mode="after")
    def _check_shape(self) -> "Question":
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: categoria sconosciuta {self.category!r}")
        if self.answerable and not (self.relevant and self.key_facts):
            raise ValueError(f"{self.id}: servono relevant e key_facts")
        if not self.answerable and (self.relevant or self.key_facts):
            raise ValueError(f"{self.id}: una domanda senza risposta non ha gold")
        return self


class Dataset(BaseModel):
    name: str
    repo_url: str
    ref: str
    questions: List[Question]


def check_name(name: str) -> str:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Nome non valido: {name!r}")
    return name


def load_dataset(name: str) -> Dataset:
    folder = DATASETS_DIR / check_name(name)
    meta = yaml.safe_load((folder / "dataset.yaml").read_text(encoding="utf-8"))
    questions = yaml.safe_load((folder / "questions.yaml").read_text(encoding="utf-8"))
    dataset = Dataset(name=name, questions=questions, **meta)
    ids = [q.id for q in dataset.questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Id domanda duplicati nel dataset")
    return dataset


def fetch_repo(dataset: Dataset) -> tuple[Path, str]:
    """Clona il repo al ref indicato (con cache) e restituisce (cartella, commit)."""
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
    """File indicizzabili (percorsi relativi alla radice, come li vede esperto)."""
    return sorted(
        path.relative_to(repo_dir).as_posix()
        for path in repo_dir.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(repo_dir).parts
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_zip(repo_dir: Path, files: list[str]) -> bytes:
    """ZIP con i file alla radice: i `filename` indicizzati coincidono con il gold."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(repo_dir / name, arcname=name)
    return buffer.getvalue()


def _top_level_symbols(path: Path) -> Optional[set[str]]:
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
    """Errori del gold set rispetto al checkout (file o simboli inesistenti)."""
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
