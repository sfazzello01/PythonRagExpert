"""Config per il frontend"""

import os
from pathlib import Path

# Frontend root directory
FRONTEND_ROOT = Path(".")

# Internal paths (relative to frontend)
FRONTEND_SRC_DIR = FRONTEND_ROOT / "src"
FRONTEND_TEMPLATES_DIR = FRONTEND_SRC_DIR / "templates"
LOGS_DIR = FRONTEND_ROOT / "logs"
LOGS_FILE = LOGS_DIR / "frontend.log"

# External services
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8003"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
