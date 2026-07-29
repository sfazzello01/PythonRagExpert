# config.py
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable using common textual values."""
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated environment variable as a normalized tuple."""
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


class AppConfig(BaseModel):
    """Configuration for Esperto RAG microservice with validation."""
    model_config = {"extra": "ignore"}
    
    # Ollama LLM Configuration
    ollama_url: str = Field(default="http://ollama:11434", description="Ollama API URL")
    llm_model: str = Field(default="llama3.2:3b", description="LLM model name in Ollama")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature (0=deterministic)")
    
    # Qdrant Vector DB Configuration
    qdrant_host: str = Field(default="qdrant", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection: str = Field(default="codebase_expert_bgem3_v1", description="Qdrant collection name")
    
    # Retrieval Hyperparameters
    k: int = Field(default=10, ge=1, le=50, description="Number of primary chunks to retrieve")
    prefetch_limit: int = Field(default=20, ge=1, le=100, description="Prefetch limit for dense+sparse before rerank")
    semantic_relevance_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="ColBERT score threshold")
    embedding_max_length: int = Field(default=1024, ge=128, le=8192, description="Maximum tokens per BGE-M3 embedding")
    
    # API Configuration
    max_message_length: int = Field(default=5000, ge=100, le=50000, description="Max message length")
    min_message_length: int = Field(default=3, ge=1, le=100, description="Min message length")
    
    # Timeout Configuration (seconds)
    model_load_timeout: float = Field(default=60.0, ge=10.0, description="Timeout for BGE-M3 model load")
    qdrant_query_timeout: float = Field(default=30.0, ge=5.0, description="Timeout for Qdrant queries")
    llm_inference_timeout: float = Field(default=120.0, ge=10.0, description="Timeout for LLM inference")

    # Indexing Configuration
    base_dir: str = Field(default="/app", description="Root directory containing indexed sources")
    bge_m3_device: str = Field(default="cuda", description="BGE-M3 device requested for indexing")
    bge_m3_use_fp16: bool = Field(default=True, description="Use FP16 for BGE-M3 indexing on CUDA")
    batch_size: int = Field(default=2, ge=1, le=128, description="Documents encoded per Qdrant upsert batch")
    index_state_dir: str = Field(default="/app/state", description="Directory for persistent index state")


def _load_config_from_env() -> AppConfig:
    """Load config from environment variables with fallback to defaults."""
    config_data = {
        "ollama_url": os.getenv("OLLAMA_URL", "http://ollama:11434"),
        "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
        "temperature": float(os.getenv("TEMPERATURE", "0.0")),
        "qdrant_host": os.getenv("QDRANT_HOST", "qdrant"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", "codebase_expert_bgem3_v1"),
        "k": int(os.getenv("K", "10")),
        "prefetch_limit": int(os.getenv("PREFETCH_LIMIT", "20")),
        "semantic_relevance_threshold": float(os.getenv("SEMANTIC_RELEVANCE_THRESHOLD", "0.0")),
        "embedding_max_length": int(os.getenv("EMBEDDING_MAX_LENGTH", "1024")),
        "max_message_length": int(os.getenv("MAX_MESSAGE_LENGTH", "5000")),
        "min_message_length": int(os.getenv("MIN_MESSAGE_LENGTH", "3")),
        "model_load_timeout": float(os.getenv("MODEL_LOAD_TIMEOUT", "60.0")),
        "qdrant_query_timeout": float(os.getenv("QDRANT_QUERY_TIMEOUT", "30.0")),
        "llm_inference_timeout": float(os.getenv("LLM_INFERENCE_TIMEOUT", "120.0")),
        "base_dir": os.getenv("BASE_DIR", "/app"),
        "bge_m3_device": os.getenv("BGE_M3_DEVICE", "cuda"),
        "bge_m3_use_fp16": _env_flag("BGE_M3_USE_FP16", True),
        "batch_size": int(os.getenv("BATCH_SIZE", "2")),
        "index_state_dir": os.getenv("INDEX_STATE_DIR", "/app/state"),
    }
    try:
        return AppConfig(**config_data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


# Load and validate config
config = _load_config_from_env()

# Expose simplified interface for backward compatibility
EMBEDDING_MODEL = None  # Removed: actually using BGE-M3 from FlagEmbedding
OLLAMA_URL = config.ollama_url
LLM_MODEL = config.llm_model
TEMPERATURE = config.temperature
QDRANT_HOST = config.qdrant_host
QDRANT_PORT = config.qdrant_port
QDRANT_COLLECTION = config.qdrant_collection
K = config.k
PREFETCH_LIMIT = config.prefetch_limit
SEMANTIC_RELEVANCE_THRESHOLD = config.semantic_relevance_threshold
EMBEDDING_MAX_LENGTH = config.embedding_max_length
MAX_MESSAGE_LENGTH = config.max_message_length
MIN_MESSAGE_LENGTH = config.min_message_length
MODEL_LOAD_TIMEOUT = config.model_load_timeout
QDRANT_QUERY_TIMEOUT = config.qdrant_query_timeout
LLM_INFERENCE_TIMEOUT = config.llm_inference_timeout

BASE_DIR = Path(config.base_dir)
BGE_M3_DEVICE = config.bge_m3_device
BGE_M3_USE_FP16 = config.bge_m3_use_fp16
BATCH_SIZE = config.batch_size
INDEX_STATE_DIR = Path(config.index_state_dir)
INDEX_STATE_FILE = INDEX_STATE_DIR / "index_state.json"
INDEX_COMPLETE_FILE = INDEX_STATE_DIR / "index_complete"

SUPPORTED_EXTENSIONS = frozenset(_env_csv(
    "SUPPORTED_EXTENSIONS",
    ".py,.json,.yaml,.yml,.md,.txt,.html,.ts,.js",
))
SKIP_DIRS = frozenset(_env_csv(
    "SKIP_DIRS",
    "__pycache__,.git,.pytest_cache,.mypy_cache,logs,mariadb_data,ollama_data,qdrant_data,node_modules,dist,build",
))
INDEX_TARGET_PATHS = _env_csv(
    "INDEX_TARGET_PATHS",
    "backend,frontend,domains.json,docker-compose.yaml",
)
PYTHON_AST_MAX_CHARS = 1500

# Chunk size ridotto per evitare l'offloading in CPU e velocizzare la KV Cache
INDEX_CHUNK_SIZE = 1000  
INDEX_CHUNK_OVERLAP = 200  
INDEX_MAX_FILE_CHARS = 200_000