import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AppConfig(BaseModel):
    """Validated application settings read from environment variables."""

    model_config = ConfigDict(
        alias_generator=str.upper,
        populate_by_name=True,
        extra="ignore",
    )

    ollama_url: str
    llm_model: str
    temperature: float = Field(ge=0.0, le=2.0)
    qdrant_host: str
    qdrant_port: int
    qdrant_collection: str
    k: int = Field(ge=1, le=50)
    semantic_relevance_ratio: float = Field(ge=0.0, le=1.0)
    embedding_max_length: int = Field(ge=128, le=8192)
    max_message_length: int = Field(ge=100, le=50000)
    min_message_length: int = Field(ge=1, le=100)
    qdrant_query_timeout: float = Field(ge=5.0)
    llm_inference_timeout: float = Field(ge=10.0)
    base_dir: Path
    batch_size: int = Field(ge=1, le=128)
    index_state_dir: Path
    dense_model_name: str = "BAAI/bge-m3"
    # True: modello quantizzato a 8 bit (piu' veloce senza OpenVINO, vettori leggermente diversi).
    embedding_int8: bool = False
    # True: l'indexer codifica i chunk su GPU NVIDIA (PyTorch, fp16); richiede l'immagine costruita con INDEXER_GPU=1.
    indexer_gpu: bool = False
    symbol_graph_max_neighbors: int = Field(default=6, ge=1)
    symbol_graph_max_depth: int = Field(default=3, ge=0)
    symbol_graph_pagerank_damping: float = Field(default=0.4, gt=0.0, lt=1.0)
    symbol_graph_pagerank_iterations: int = Field(default=20, ge=1)
    symbol_graph_pagerank_tolerance: float = Field(default=1e-6, gt=0.0)
    # Finestra del modello (num_ctx di Ollama): da qui deriva il budget del contesto RAG.
    llm_context_tokens: int = Field(default=20_000, ge=2_000)
    rag_context_max_sources: int = Field(default=35, ge=1)
    index_chunk_size: int = Field(default=1200, ge=1)
    index_chunk_overlap: int = Field(default=150, ge=0)
    supported_extensions: frozenset[str]
    skip_dirs: frozenset[str]
    max_upload_mb: int = Field(default=200, ge=1)
    max_extracted_mb: int = Field(default=1000, ge=1)
    ingestion_workspace_dir: Path = Path("/app/workspace")

    @field_validator("supported_extensions", "skip_dirs", mode="before")
    @classmethod
    def parse_csv(cls, value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(value, str):
            value = value.split(",")
        return tuple(item.strip() for item in value if item.strip())


def _load_config_from_env() -> AppConfig:
    try:
        return AppConfig.model_validate(os.environ)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


# Load and validate config
config = _load_config_from_env()

OLLAMA_URL = config.ollama_url
LLM_MODEL = config.llm_model
TEMPERATURE = config.temperature
QDRANT_HOST = config.qdrant_host
QDRANT_PORT = config.qdrant_port
QDRANT_COLLECTION = config.qdrant_collection
K = config.k
SEMANTIC_RELEVANCE_RATIO = config.semantic_relevance_ratio
EMBEDDING_MAX_LENGTH = config.embedding_max_length
MAX_MESSAGE_LENGTH = config.max_message_length
MIN_MESSAGE_LENGTH = config.min_message_length
QDRANT_QUERY_TIMEOUT = config.qdrant_query_timeout
LLM_INFERENCE_TIMEOUT = config.llm_inference_timeout

BASE_DIR = config.base_dir
BATCH_SIZE = config.batch_size
INDEX_STATE_DIR = config.index_state_dir
DENSE_MODEL_NAME = config.dense_model_name
EMBEDDING_INT8 = config.embedding_int8
INDEXER_GPU = config.indexer_gpu
# Identifica i vettori prodotti: entra nell'impronta dell'indice e nella chiave della cache.
EMBEDDING_MODEL_ID = f"{DENSE_MODEL_NAME}:int8" if EMBEDDING_INT8 else DENSE_MODEL_NAME
SYMBOL_GRAPH_MAX_NEIGHBORS = config.symbol_graph_max_neighbors
SYMBOL_GRAPH_MAX_DEPTH = config.symbol_graph_max_depth
SYMBOL_GRAPH_PAGERANK_DAMPING = config.symbol_graph_pagerank_damping
SYMBOL_GRAPH_PAGERANK_ITERATIONS = config.symbol_graph_pagerank_iterations
SYMBOL_GRAPH_PAGERANK_TOLERANCE = config.symbol_graph_pagerank_tolerance
LLM_CONTEXT_TOKENS = config.llm_context_tokens
# Due terzi della finestra al contesto (~3 caratteri per token di codice), il resto a
# istruzioni e risposta: oltre la finestra Ollama tronca l'inizio del prompt.
RAG_CONTEXT_MAX_CHARS = LLM_CONTEXT_TOKENS * 2 // 3 * 3
# Mappa del repository (README, cartelle, gerarchia delle classi) inserita in ogni prompt prima del contesto.
REPO_MAP_MAX_CHARS = RAG_CONTEXT_MAX_CHARS * 15 // 100
RAG_CONTEXT_MAX_SOURCES = config.rag_context_max_sources
INDEX_CHUNK_SIZE = config.index_chunk_size
INDEX_CHUNK_OVERLAP = config.index_chunk_overlap
INDEX_STATE_FILE = INDEX_STATE_DIR / "index_state.json"
INDEX_COMPLETE_FILE = INDEX_STATE_DIR / "index_complete"
SYMBOL_GRAPH_FILE = INDEX_STATE_DIR / "symbol_graph.json"
REPO_MAP_FILE = INDEX_STATE_DIR / "repo_map.txt"
EMBEDDING_CACHE_FILE = INDEX_STATE_DIR / "embedding_cache.sqlite"

SUPPORTED_EXTENSIONS = config.supported_extensions
SKIP_DIRS = config.skip_dirs
MAX_UPLOAD_MB = config.max_upload_mb
MAX_EXTRACTED_MB = config.max_extracted_mb
INGESTION_WORKSPACE_DIR = config.ingestion_workspace_dir

DENSE_DIMENSION = 1024
SYMBOL_GRAPH_VERSION = 1
# Da incrementare quando cambiano il chunking, i vettori salvati o la mappa: invalida gli indici gia' costruiti.
CHUNKING_VERSION = 9
