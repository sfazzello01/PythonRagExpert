"""Pytest bootstrap: ambiente minimo per importare la config del servizio.

La config dell'esperto e' risolta a import-time da variabili d'ambiente
obbligatorie: qui ne impostiamo valori fittizi prima di qualunque import
dei moduli sotto test. Nessun test tocca servizi esterni (Ollama/Qdrant).
"""

import os
import sys
from pathlib import Path

ESPERTO_ROOT = Path(__file__).resolve().parents[1]
if str(ESPERTO_ROOT) not in sys.path:
    sys.path.insert(0, str(ESPERTO_ROOT))

_DEFAULT_ENV = {
    "OLLAMA_URL": "http://localhost:11434",
    "LLM_MODEL": "test-model",
    "TEMPERATURE": "0.1",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_COLLECTION": "test_collection",
    "K": "8",
    "PREFETCH_LIMIT": "20",
    "SEMANTIC_RELEVANCE_THRESHOLD": "0.3",
    "EMBEDDING_MAX_LENGTH": "1024",
    "MAX_MESSAGE_LENGTH": "5000",
    "MIN_MESSAGE_LENGTH": "3",
    "MODEL_LOAD_TIMEOUT": "60",
    "QDRANT_QUERY_TIMEOUT": "30",
    "LLM_INFERENCE_TIMEOUT": "300",
    "BASE_DIR": "/tmp/esperto-test-base",
    "BGE_M3_DEVICE": "cpu",
    "BGE_M3_USE_FP16": "false",
    "BATCH_SIZE": "4",
    "INDEX_STATE_DIR": "/tmp/esperto-test-state",
    "SUPPORTED_EXTENSIONS": ".py,.yaml,.yml,.md,.txt,.html,.json,.js",
    "SKIP_DIRS": "__pycache__,.git",
    "INDEX_TARGET_PATHS": ".",
}

for key, value in _DEFAULT_ENV.items():
    os.environ.setdefault(key, value)
