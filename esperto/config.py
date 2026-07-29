# config.py
import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional


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

SUPPORTED_EXTENSIONS = (".py", ".json", ".txt", ".sh", ".html", ".css")
PYTHON_AST_MAX_CHARS = 1500

# Chunk size ridotto per evitare l'offloading in CPU e velocizzare la KV Cache
INDEX_CHUNK_SIZE = 1000  
INDEX_CHUNK_OVERLAP = 200  
INDEX_MAX_FILE_CHARS = 200_000