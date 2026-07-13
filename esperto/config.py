"""config per l'esperto"""

EMBEDDING_MODEL = "nomic-embed-text:latest"  # Sostituisci con il nome del tuo modello di embedding su Ollama
OLLAMA_URL = "http://ollama:11434"
QDRANT_HOST = "qdrant"
QDRANT_COLLECTION = "codebase_expert"
LLM_MODEL = "llama3.2:3b"
TEMPERATURE = 0.0  # temperatura per risposte più deterministiche, da finetunare
K = 25  # numero di documenti da recuperare, aumentato per domande generali
SUPPORTED_EXTENSIONS = (".py", ".json", ".txt", ".sh", ".html", ".css")
PYTHON_AST_MAX_CHARS = 1500
INDEX_CHUNK_SIZE = 2000  # aumentato per mantenere più contesto per chunk
INDEX_CHUNK_OVERLAP = 400  # aumentato per migliore continuità tra chunk
INDEX_MAX_FILE_CHARS = 200_000
