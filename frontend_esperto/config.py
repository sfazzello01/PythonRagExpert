"""Configurazione del frontend dell'esperto: percorsi interni e URL dei servizi."""

import os
from pathlib import Path

# Radice del frontend.
FRONTEND_ROOT = Path(".")

# Percorsi interni, relativi alla radice del frontend.
FRONTEND_SRC_DIR = FRONTEND_ROOT / "src"
FRONTEND_TEMPLATES_DIR = FRONTEND_SRC_DIR / "templates"
FRONTEND_STATIC_DIR = FRONTEND_SRC_DIR / "static"
LOGS_DIR = FRONTEND_ROOT / "logs"
LOGS_FILE = LOGS_DIR / "frontend_esperto.log"

# Servizi esterni (stack dell'esperto); gli URL sono senza barra finale.
ESPERTO_HOST = os.getenv("ESPERTO_HOST", "127.0.0.1")
ESPERTO_PORT = int(os.getenv("ESPERTO_PORT", "5000"))
ESPERTO_URL = os.getenv("ESPERTO_URL", f"http://{ESPERTO_HOST}:{ESPERTO_PORT}").rstrip("/")

ESPERTO_INDEXER_HOST = os.getenv("ESPERTO_INDEXER_HOST", "127.0.0.1")
ESPERTO_INDEXER_PORT = int(os.getenv("ESPERTO_INDEXER_PORT", "5001"))
ESPERTO_INDEXER_URL = os.getenv(
    "ESPERTO_INDEXER_URL", f"http://{ESPERTO_INDEXER_HOST}:{ESPERTO_INDEXER_PORT}"
).rstrip("/")
