"""Config per il backend."""

from pathlib import Path

# Backend root directory
BACKEND_ROOT = Path(".")

# Internal paths
BACKEND_DIR = BACKEND_ROOT
BACKEND_GS_HTML_DIR = BACKEND_ROOT / "gs_html"
CRAWLED_DIR = BACKEND_ROOT / "crawled"
LOGS_DIR = BACKEND_ROOT / "logs"
LOGS_FILE = LOGS_DIR / "backend.log"

# External data
GS_DATA_DIR = BACKEND_ROOT / "gs_data"
DOMAINS_FILE = BACKEND_ROOT / "domains.json"

#modelli llm
GEMMA = "gemma4:e2b"
LLAMA = "llama3.2:3b"
PHI = "phi4-mini"
QWEN = "qwen3:4b"
MINISTRAL = "ministral-3:3b"

MODEL = LLAMA #PHI per lo sviluppo perchè è velocissimo, GEMMA per la produzione

THINK = False #False per lo sviluppo, True per la produzione

NO_LLM = False #True per lo sviluppo, False per la produzione
