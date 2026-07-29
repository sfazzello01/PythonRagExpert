from __future__ import annotations

import os
import re
import ast
import time
import uuid
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
    MultiVectorConfig,
    MultiVectorComparator,
)
from qdrant_client.http.exceptions import UnexpectedResponse
from FlagEmbedding import BGEM3FlagModel
import torch

# =====================================================================
# CONFIGURAZIONE E VARIABILI D'AMBIENTE
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EspertoSOTA")

BASE_DIR = Path(os.getenv("BASE_DIR", "/app"))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "codebase_expert_bgem3_v1")

# Modello BGE-M3 unificato (PyTorch via FlagEmbedding, non ONNX/fastembed):
# un solo forward pass produce dense + sparse (lexical_weights) + ColBERT (multi-vettore)
DENSE_MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024
BGE_M3_DEVICE = os.getenv("BGE_M3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if BGE_M3_DEVICE.startswith("cuda") and not torch.cuda.is_available():
    logger.warning("CUDA richiesta ma non disponibile: uso CPU per BGE-M3.")
    BGE_M3_DEVICE = "cpu"
BGE_M3_USE_FP16 = (
    os.getenv("BGE_M3_USE_FP16", "true").lower() == "true"
    and BGE_M3_DEVICE.startswith("cuda")
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "1024"))
INDEX_STATE_DIR = Path(os.getenv("INDEX_STATE_DIR", str(BASE_DIR / "state")))
INDEX_STATE_FILE = INDEX_STATE_DIR / "index_state.json"
INDEX_COMPLETE_FILE = INDEX_STATE_DIR / "index_complete"

SUPPORTED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".html", ".ts", ".js"}
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", "logs", 
    "mariadb_data", "ollama_data", "qdrant_data", "node_modules", "dist", "build"
}
INDEX_TARGETS = (
    BASE_DIR / "backend",
    BASE_DIR / "frontend",
    BASE_DIR / "config.py",
    BASE_DIR / "domains.json",
    BASE_DIR / "docker-compose.yaml",
)


# =====================================================================
# 1. FUNZIONAMENTO AST CHUNKER & CONTEXTUAL ENRICHMENT
# =====================================================================
class PythonASTContextualChunker:
    """Estrae funzioni e classi da codice Python, iniettando nel contesto

    gli import del modulo, le route FastAPI e le firme dei decoratori.
    """
    def __init__(self, file_path: Path, code: str):
        self.file_path = file_path
        self.relative_path = str(file_path.relative_to(BASE_DIR)) if BASE_DIR in file_path.parents else str(file_path)
        self.code = code
        self.lines = code.splitlines(keepends=True)

    def extract_chunks(self) -> List[Document]:
        documents = []
        try:
            tree = ast.parse(self.code)
        except SyntaxError as e:
            logger.warning(f"⚠️ Errore sintassi in {self.relative_path}: {e}. Fallback su splitter grezzo.")
            return _fallback_text_split(self.file_path, self.code, language="python")

        # Estrazione contesto globale del modulo
        imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        imports_context = "\n".join(imports) + "\n\n" if imports else ""
        processed_lines = set()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                for l in range(start, end):
                    processed_lines.add(l)

                raw_code = "".join(self.lines[start:end])
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                
                # Identificazione route FastAPI
                route_meta = ""
                for dec in getattr(node, "decorator_list", []):
                    dec_str = ast.unparse(dec)
                    if any(m in dec_str for m in ["get(", "post(", "put(", "delete(", "patch("]):
                        route_meta = f"# ENDPOINT API: {dec_str}\n"

                # CONTEXTUAL RETRIEVAL: Prepariamo un'intestazione ricca per il modello bge-m3
                enrichment_header = (
                    f"# FILE: {self.relative_path} | SERVIZIO: {_service_for_path(self.file_path)}\n"
                    f"# TIPO: {chunk_type.upper()} | NOME: {node.name}\n"
                    f"{route_meta}"
                    f"--- IMPORTS DI CONTESTO ---\n{imports_context}"
                    f"--- CODICE SORGENTE ---\n"
                )

                content = enrichment_header + raw_code
                doc_id = hashlib.md5(f"{self.relative_path}::{node.name}::{start}::{end}".encode('utf-8')).hexdigest()

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "filename": self.relative_path,
                        "service": _service_for_path(self.file_path),
                        "chunk_type": chunk_type,
                        "symbol_name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "chunk_id": doc_id
                    }
                ))

        # Cattura eventuale codice di primo livello (variabili globali, config, script)
        global_lines = [self.lines[i] for i in range(len(self.lines)) if i not in processed_lines and self.lines[i].strip()]
        if global_lines and len("".join(global_lines).strip()) > 50:
            content = f"# FILE: {self.relative_path} (Global Body)\n\n" + "".join(global_lines)
            doc_id = hashlib.md5(f"{self.relative_path}::global_body".encode('utf-8')).hexdigest()
            documents.append(Document(
                page_content=content,
                metadata={
                    "filename": self.relative_path,
                    "service": _service_for_path(self.file_path),
                    "chunk_type": "global_body",
                    "symbol_name": "main_body",
                    "chunk_id": doc_id
                }
            ))

        return documents


def _fallback_text_split(path: Path, text: str, language: str = "text") -> List[Document]:
    """Splitter generico per file non-Python (YAML, Markdown, Frontend, JSON lunghi)."""
    relative_path = str(path.relative_to(BASE_DIR)) if BASE_DIR in path.parents else str(path)
    service = _service_for_path(path)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    raw_chunks = splitter.split_text(text)
    
    docs = []
    for i, chunk in enumerate(raw_chunks):
        content = f"# FILE: {relative_path} (Part {i+1}/{len(raw_chunks)}) | SERVIZIO: {service}\n\n{chunk}"
        doc_id = hashlib.md5(f"{relative_path}::part_{i}".encode('utf-8')).hexdigest()
        docs.append(Document(
            page_content=content,
            metadata={
                "filename": relative_path,
                "service": service,
                "chunk_type": "text_block",
                "symbol_name": f"part_{i+1}",
                "chunk_id": doc_id
            }
        ))
    return docs


# =====================================================================
# 2. GENERAZIONE DELLA MAPPA GLOBALE ARCHITETTURALE
# =====================================================================
def _extract_fastapi_routes(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in getattr(node, "decorator_list", []):
                dec_str = ast.unparse(decorator)
                if any(m in dec_str for m in ["get(", "post(", "put(", "delete(", "patch("]):
                    routes.append(f"{dec_str} -> {node.name}()")
    return routes


def _extract_frontend_fetches(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    fetches = []
    for match in re.finditer(r"fetch\s*\(\s*([^,\n)]+)", text):
        endpoint = match.group(1).strip().strip("'`\"")
        line = text.count("\n", 0, match.start()) + 1
        fetches.append(f"linea {line}: fetch({endpoint})")
    return fetches


def _generate_target_tree(path: Path, prefix: str = "") -> list[str]:
    lines = []
    try:
        items = sorted([i for i in path.iterdir() if i.name not in SKIP_DIRS], key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                lines.extend(_generate_target_tree(item, prefix + ("    " if is_last else "│   ")))
            elif item.suffix in SUPPORTED_EXTENSIONS or item.name in ["docker-compose.yaml", "domains.json"]:
                lines.append(f"{prefix}{connector}{item.name}")
    except OSError:
        pass
    return lines


def _build_codebase_summary_documents(all_files: List[Path]) -> list[Document]:
    """Genera il documento di mappa globale '__codebase_map__' per le query architetturali."""
    logger.info("🗺️ Generazione documento overview '__codebase_map__'")
    sections = []
    total_py, total_routes, total_fetches = 0, 0, 0

    for path in all_files:
        rel_path = str(path.relative_to(BASE_DIR)) if BASE_DIR in path.parents else str(path)
        details = []
        if path.suffix == ".py":
            total_py += 1
            routes = _extract_fastapi_routes(path)
            if routes:
                details.append("ROUTE FASTAPI:\n  " + "\n  ".join(routes))
                total_routes += len(routes)
        if path.suffix in {".html", ".py", ".ts", ".js"}:
            fetches = _extract_frontend_fetches(path)
            if fetches:
                details.append("CHIAMATE HTTP/FETCH:\n  " + "\n  ".join(fetches))
                total_fetches += len(fetches)
        if details:
            sections.append(f"FILE: {rel_path}\n" + "\n".join(details))

    tree_lines = []
    for target in INDEX_TARGETS:
        if target.exists():
            tree_lines.append(f"└── {target.name}/" if target.is_dir() else f"└── {target.name}")
            if target.is_dir():
                tree_lines.extend(_generate_target_tree(target, "    "))

    overview_text = (
        "==================================================================\n"
        "             OVERVIEW GENERALE DELLA ARCHITETTURA RAG\n"
        "==================================================================\n\n"
        f"METRICHE CHIAVE:\n- File indicizzati: {len(all_files)}\n- Moduli Python: {total_py}\n"
        f"- Endpoint API esposti: {total_routes}\n- Chiamate Fetch Frontend: {total_fetches}\n\n"
        "ALBERO DELLE CARTELLE:\n" + "\n".join(tree_lines) + "\n\n"
        "==================================================================\n"
        "                 MAPPA ROUTE E CHIAMATE DEI FILE\n"
        "==================================================================\n\n" +
        "\n\n".join(sections)
    )

    doc_id = hashlib.md5("global_codebase_map_sota".encode('utf-8')).hexdigest()
    return [Document(
        page_content=overview_text,
        metadata={"filename": "__codebase_map__", "service": "architecture", "chunk_type": "architecture_map", "chunk_id": doc_id}
    )]


# =====================================================================
# 3. UTILITY DI SISTEMA E SCANSIONE
# =====================================================================
def _service_for_path(path: Path) -> str:
    try:
        rel = path.relative_to(BASE_DIR)
        return rel.parts[0] if rel.parts else "root"
    except ValueError:
        return "root"


def _iter_source_files() -> list[Path]:
    files = []
    for target in INDEX_TARGETS:
        if target.is_file() and target.suffix in SUPPORTED_EXTENSIONS:
            files.append(target)
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS and not any(p in SKIP_DIRS for p in path.parts):
                    files.append(path)
    return sorted(files)


def _source_fingerprint(files: list[Path]) -> str:
    """Return a stable fingerprint of indexed sources and embedding settings."""
    digest = hashlib.sha256()
    digest.update(f"{DENSE_MODEL_NAME}:{DENSE_DIM}:{EMBEDDING_MAX_LENGTH}".encode("utf-8"))
    for path in files:
        relative_path = path.relative_to(BASE_DIR).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _index_is_current(fingerprint: str) -> bool:
    try:
        state = json.loads(INDEX_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        state.get("collection") == QDRANT_COLLECTION
        and state.get("fingerprint") == fingerprint
    )


def _clear_index_state() -> None:
    INDEX_STATE_FILE.unlink(missing_ok=True)
    INDEX_COMPLETE_FILE.unlink(missing_ok=True)


def _save_index_state(fingerprint: str, source_file_count: int) -> None:
    INDEX_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "collection": QDRANT_COLLECTION,
        "fingerprint": fingerprint,
        "source_file_count": source_file_count,
    }
    temporary_file = INDEX_STATE_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_file.replace(INDEX_STATE_FILE)
    INDEX_COMPLETE_FILE.touch()


def _wait_for_qdrant() -> QdrantClient:
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    logger.info(f"🔌 Connessione a Qdrant su {url}...")
    client = QdrantClient(url=url)
    for attempt in range(1, 31):
        try:
            client.get_collections()
            logger.info("✅ Qdrant è online e risponde.")
            return client
        except Exception as e:
            logger.warning(f"⏳ Attesa Qdrant (Tentativo {attempt}/30): {e}")
            time.sleep(2)
    raise RuntimeError("❌ Qdrant non disponibile dopo 60 secondi.")


# =====================================================================
# 4. ENGINE MOTORE IBRIDO QDRANT (DENSE + SPARSE + COLBERT, TUTTI DA BGE-M3)
# =====================================================================
def load_bge_m3_model() -> BGEM3FlagModel:
    """Carica BGE-M3 in locale (PyTorch/FlagEmbedding).

    A differenza di fastembed (ONNX), un solo forward pass restituisce
    dense_vecs, lexical_weights (sparse) e colbert_vecs: qui sfruttiamo
    davvero la natura "ibrida" del modello, non solo il suo embedding denso.
    """
    logger.info(
        f"🧠 Caricamento modello BGE-M3 unificato "
        f"({DENSE_MODEL_NAME}, device={BGE_M3_DEVICE}, fp16={BGE_M3_USE_FP16})..."
    )
    return BGEM3FlagModel(
        DENSE_MODEL_NAME,
        use_fp16=BGE_M3_USE_FP16,
        devices=BGE_M3_DEVICE,
    )


def ensure_hybrid_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the hybrid collection, optionally replacing an outdated index."""
    if client.collection_exists(QDRANT_COLLECTION):
        if not recreate:
            logger.info(f"📦 Collection '{QDRANT_COLLECTION}' trovata.")
            return
        logger.info(f"♻️ Rimozione collection obsoleta '{QDRANT_COLLECTION}'.")
        client.delete_collection(QDRANT_COLLECTION)

    logger.info(f"✨ Creazione nuova collection ibrida '{QDRANT_COLLECTION}' (dense + sparse + colbert)...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            # ColBERT: un vettore per token, comparati con MaxSim invece di un singolo cosine
            "colbert": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            ),
        },
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Converte l'MD5 esadecimale (32 char) già usato come chunk_id in un UUID valido per Qdrant.

    Un digest MD5 ha esattamente la stessa lunghezza esadecimale di un UUID (32 char),
    quindi la conversione è deterministica e biunivoca: stesso contenuto -> stesso ID,
    preservando l'idempotenza dell'ingestione incrementale (niente force_recreate).
    """
    return str(uuid.UUID(hex=chunk_id))


def _lexical_weights_to_sparse_vector(lexical_weights: Dict[str, float]) -> SparseVector:
    if not lexical_weights:
        return SparseVector(indices=[], values=[])
    indices, values = zip(*((int(token_id), float(weight)) for token_id, weight in lexical_weights.items()))
    return SparseVector(indices=list(indices), values=list(values))


def encode_and_upsert_batch(model: BGEM3FlagModel, client: QdrantClient, batch: List[Document]) -> None:
    """Genera dense+sparse+colbert per un batch di chunk in un solo forward pass e li carica su Qdrant."""
    texts = [doc.page_content for doc in batch]

    output = model.encode(
        texts,
        batch_size=len(texts),
        max_length=EMBEDDING_MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    points = []
    for doc, dense_vec, lexical_weight, colbert_vec in zip(
        batch, output["dense_vecs"], output["lexical_weights"], output["colbert_vecs"]
    ):
        points.append(
            PointStruct(
                id=_chunk_id_to_point_id(doc.metadata["chunk_id"]),
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": _lexical_weights_to_sparse_vector(lexical_weight),
                    "colbert": colbert_vec.tolist(),
                },
                payload={"page_content": doc.page_content, "metadata": doc.metadata},
            )
        )

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    except UnexpectedResponse as error:
        if "larger than allowed" not in str(error) or len(points) == 1:
            raise

        logger.warning(
            "Payload batch troppo grande per Qdrant; eseguo l'upsert dei %d chunk singolarmente.",
            len(points),
        )
        for point in points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=[point])


def build_index() -> None:
    """Funzione principale di orchestrazione della pipeline SOTA."""
    start_time = time.time()
    client = _wait_for_qdrant()

    all_files = _iter_source_files()
    logger.info(f"📂 Scansione completata: trovati {len(all_files)} file validi da processare.")
    fingerprint = _source_fingerprint(all_files)

    if client.collection_exists(QDRANT_COLLECTION) and _index_is_current(fingerprint):
        INDEX_COMPLETE_FILE.touch()
        logger.info("✅ Codebase invariata: indice persistente già aggiornato, indicizzazione saltata.")
        return

    _clear_index_state()
    ensure_hybrid_collection(client, recreate=client.collection_exists(QDRANT_COLLECTION))
    model = load_bge_m3_model()

    all_documents: List[Document] = []

    # 1. Chunking semantico file per file
    for path in all_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
                
            if path.suffix == ".py":
                chunker = PythonASTContextualChunker(path, text)
                all_documents.extend(chunker.extract_chunks())
            else:
                all_documents.extend(_fallback_text_split(path, text, language=path.suffix.lstrip(".")))
        except Exception as e:
            logger.error(f"❌ Errore processamento {path}: {e}")

    # 2. Aggiunta mappa architetturale globale
    all_documents.extend(_build_codebase_summary_documents(all_files))

    # 3. Batching e Ingestione Idempotente (senza force_recreate!)
    total_docs = len(all_documents)
    logger.info(f"🚀 Avvio indicizzazione ibrida di {total_docs} chunk semantici su Qdrant...")

    for i in range(0, total_docs, BATCH_SIZE):
        batch = all_documents[i : i + BATCH_SIZE]
        batch_start_time = time.time()

        # encode_and_upsert_batch aggiorna solo i chunk con ID (derivato da MD5) modificato o nuovo
        encode_and_upsert_batch(model, client, batch)
        batch_elapsed = time.time() - batch_start_time
        logger.info(
            f"   ⏳ Caricati e indicizzati {min(i + BATCH_SIZE, total_docs)}/{total_docs} chunk "
            f"in {batch_elapsed:.1f}s..."
        )

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"🎯 Indicizzazione SOTA completata con successo in {elapsed}s! Database pronto per query ibride.")

    _save_index_state(fingerprint, len(all_files))
    logger.info(f"🏁 Stato persistente creato: {INDEX_STATE_FILE}")


if __name__ == "__main__":
    build_index()