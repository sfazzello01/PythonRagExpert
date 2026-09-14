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
    prefetch_limit: int = Field(ge=1, le=100)
    semantic_relevance_ratio: float = Field(ge=0.0, le=1.0)
    embedding_max_length: int = Field(ge=128, le=8192)
    max_message_length: int = Field(ge=100, le=50000)
    min_message_length: int = Field(ge=1, le=100)
    model_load_timeout: float = Field(ge=10.0)
    qdrant_query_timeout: float = Field(ge=5.0)
    llm_inference_timeout: float = Field(ge=10.0)
    base_dir: Path
    bge_m3_device: str
    bge_m3_use_fp16: bool
    batch_size: int = Field(ge=1, le=128)
    index_state_dir: Path
    dense_model_name: str = "BAAI/bge-m3"
    symbol_graph_max_neighbors: int = Field(default=6, ge=1)
    symbol_graph_max_depth: int = Field(default=2, ge=0)
    rag_context_max_chars: int = Field(default=60_000, ge=1)
    rag_context_max_sources: int = Field(default=35, ge=1)
    index_chunk_size: int = Field(default=1200, ge=1)
    index_chunk_overlap: int = Field(default=150, ge=0)
    supported_extensions: frozenset[str]
    skip_dirs: frozenset[str]
    index_target_paths: tuple[str, ...]

    @field_validator("supported_extensions", "skip_dirs", "index_target_paths", mode="before")
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
PREFETCH_LIMIT = config.prefetch_limit
SEMANTIC_RELEVANCE_RATIO = config.semantic_relevance_ratio
EMBEDDING_MAX_LENGTH = config.embedding_max_length
MAX_MESSAGE_LENGTH = config.max_message_length
MIN_MESSAGE_LENGTH = config.min_message_length
MODEL_LOAD_TIMEOUT = config.model_load_timeout
QDRANT_QUERY_TIMEOUT = config.qdrant_query_timeout
LLM_INFERENCE_TIMEOUT = config.llm_inference_timeout

BASE_DIR = config.base_dir
BGE_M3_DEVICE = config.bge_m3_device
BGE_M3_USE_FP16 = config.bge_m3_use_fp16
BATCH_SIZE = config.batch_size
INDEX_STATE_DIR = config.index_state_dir
DENSE_MODEL_NAME = config.dense_model_name
SYMBOL_GRAPH_MAX_NEIGHBORS = config.symbol_graph_max_neighbors
SYMBOL_GRAPH_MAX_DEPTH = config.symbol_graph_max_depth
RAG_CONTEXT_MAX_CHARS = config.rag_context_max_chars
RAG_CONTEXT_MAX_SOURCES = config.rag_context_max_sources
INDEX_CHUNK_SIZE = config.index_chunk_size
INDEX_CHUNK_OVERLAP = config.index_chunk_overlap
INDEX_STATE_FILE = INDEX_STATE_DIR / "index_state.json"
INDEX_COMPLETE_FILE = INDEX_STATE_DIR / "index_complete"

SUPPORTED_EXTENSIONS = config.supported_extensions
SKIP_DIRS = config.skip_dirs
INDEX_TARGET_PATHS = config.index_target_paths