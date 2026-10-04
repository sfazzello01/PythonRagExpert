"""Bootstrap pytest: path del servizio e directory di lavoro fittizie (nessun servizio esterno)."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("BENCHMARK_DATASETS_DIR", str(ROOT / "datasets"))
os.environ.setdefault("BENCHMARK_RESULTS_DIR", tempfile.mkdtemp(prefix="bench-results-"))
os.environ.setdefault("BENCHMARK_REPO_CACHE_DIR", tempfile.mkdtemp(prefix="bench-repos-"))
os.environ.setdefault("LLM_MODEL", "test-model")
