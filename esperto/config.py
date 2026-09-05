# config.py
import os
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError


def _env_flag(name: str) -> bool:
    """Read a required boolean environment variable."""
    return os.environ[name].lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple[str, ...]:
    """Read a required comma-separated environment variable."""
    return tuple(value.strip() for value in os.environ[name].split(",") if value.strip())


class AppConfig(BaseModel):
    """Configuration for Esperto RAG microservice with validation."""
    model_config = {"extra": "ignore"}
    
    # Ollama LLM Configuration
    ollama_url: str = Field(description="Ollama API URL")
    llm_model: str = Field(description="LLM model name in Ollama")
    temperature: float = Field(ge=0.0, le=2.0, description="LLM temperature (0=deterministic)")
    
    # Qdrant Vector DB Configuration
    qdrant_host: str = Field(description="Qdrant host")
    qdrant_port: int = Field(description="Qdrant port")
    qdrant_collection: str = Field(description="Qdrant collection name")
    
    # Retrieval Hyperparameters
    k: int = Field(ge=1, le=50, description="Number of primary chunks to retrieve")
    prefetch_limit: int = Field(ge=1, le=100, description="Prefetch limit for dense+sparse before rerank")
    semantic_relevance_threshold: float = Field(ge=0.0, le=1.0, description="ColBERT score threshold")
    embedding_max_length: int = Field(ge=128, le=8192, description="Maximum tokens per BGE-M3 embedding")
    
    # API Configuration
    max_message_length: int = Field(ge=100, le=50000, description="Max message length")
    min_message_length: int = Field(ge=1, le=100, description="Min message length")
    
    # Timeout Configuration (seconds)
    model_load_timeout: float = Field(ge=10.0, description="Timeout for BGE-M3 model load")
    qdrant_query_timeout: float = Field(ge=5.0, description="Timeout for Qdrant queries")
    llm_inference_timeout: float = Field(ge=10.0, description="Timeout for LLM inference")

    # Indexing Configuration
    base_dir: str = Field(description="Root directory containing indexed sources")
    bge_m3_device: str = Field(description="BGE-M3 device requested for indexing")
    bge_m3_use_fp16: bool = Field(description="Use FP16 for BGE-M3 indexing on CUDA")
    batch_size: int = Field(ge=1, le=128, description="Documents encoded per Qdrant upsert batch")
    index_state_dir: str = Field(description="Directory for persistent index state")


def _load_config_from_env() -> AppConfig:
    """Load the complete application configuration from the root .env file."""
    config_data = {
        "ollama_url": os.environ["OLLAMA_URL"],
        "llm_model": os.environ["LLM_MODEL"],
        "temperature": float(os.environ["TEMPERATURE"]),
        "qdrant_host": os.environ["QDRANT_HOST"],
        "qdrant_port": int(os.environ["QDRANT_PORT"]),
        "qdrant_collection": os.environ["QDRANT_COLLECTION"],
        "k": int(os.environ["K"]),
        "prefetch_limit": int(os.environ["PREFETCH_LIMIT"]),
        "semantic_relevance_threshold": float(os.environ["SEMANTIC_RELEVANCE_THRESHOLD"]),
        "embedding_max_length": int(os.environ["EMBEDDING_MAX_LENGTH"]),
        "max_message_length": int(os.environ["MAX_MESSAGE_LENGTH"]),
        "min_message_length": int(os.environ["MIN_MESSAGE_LENGTH"]),
        "model_load_timeout": float(os.environ["MODEL_LOAD_TIMEOUT"]),
        "qdrant_query_timeout": float(os.environ["QDRANT_QUERY_TIMEOUT"]),
        "llm_inference_timeout": float(os.environ["LLM_INFERENCE_TIMEOUT"]),
        "base_dir": os.environ["BASE_DIR"],
        "bge_m3_device": os.environ["BGE_M3_DEVICE"],
        "bge_m3_use_fp16": _env_flag("BGE_M3_USE_FP16"),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "index_state_dir": os.environ["INDEX_STATE_DIR"],
    }
    try:
        return AppConfig(**config_data)
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

SUPPORTED_EXTENSIONS = frozenset(_env_csv("SUPPORTED_EXTENSIONS"))
SKIP_DIRS = frozenset(_env_csv("SKIP_DIRS"))
INDEX_TARGET_PATHS = _env_csv("INDEX_TARGET_PATHS")