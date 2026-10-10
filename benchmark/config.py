"""Configurazione del servizio di benchmark (variabili d'ambiente)."""

import os
from pathlib import Path

ESPERTO_URL = os.getenv("ESPERTO_URL", "http://esperto:5000")
INDEXER_URL = os.getenv("ESPERTO_INDEXER_URL", "http://esperto_indexer:5001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Modello di esperto, usato dalla baseline closed-book. Le risposte sono giudicate fuori dal
# servizio (vedi `to_judge.jsonl` e POST /runs/{id}/judgments): nessun LLM giudice qui.
LLM_MODEL = os.getenv("LLM_MODEL", "")

REQUEST_TIMEOUT = float(os.getenv("BENCHMARK_REQUEST_TIMEOUT", "300"))
INDEX_TIMEOUT_SECONDS = float(os.getenv("BENCHMARK_INDEX_TIMEOUT", "3600"))

DATASETS_DIR = Path(os.getenv("BENCHMARK_DATASETS_DIR", "/app/datasets"))
RESULTS_DIR = Path(os.getenv("BENCHMARK_RESULTS_DIR", "/app/results"))
REPO_CACHE_DIR = Path(os.getenv("BENCHMARK_REPO_CACHE_DIR", "/app/results/.repos"))

# Estensioni indicizzate da esperto: lo ZIP include solo queste.
SUPPORTED_EXTENSIONS = frozenset(
    ext.strip()
    for ext in os.getenv("SUPPORTED_EXTENSIONS", ".py,.yaml,.yml,.md,.txt,.html,.json,.js").split(",")
    if ext.strip()
)

# Parametri RAG registrati nel manifest di ogni run (stesso .env di esperto).
MANIFEST_ENV_KEYS = (
    "LLM_MODEL",
    "TEMPERATURE",
    "DENSE_MODEL_NAME",
    "EMBEDDING_MAX_LENGTH",
    "K",
    "SEMANTIC_RELEVANCE_RATIO",
    "INDEX_CHUNK_SIZE",
    "INDEX_CHUNK_OVERLAP",
    "SYMBOL_GRAPH_MAX_NEIGHBORS",
    "SYMBOL_GRAPH_MAX_DEPTH",
    "SYMBOL_GRAPH_PAGERANK_DAMPING",
    "LLM_CONTEXT_TOKENS",
    "RAG_CONTEXT_MAX_SOURCES",
)
