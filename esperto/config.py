"""config per l'esperto"""

EMBEDDING_MODEL = "nomic-embed-text:latest"  # Sostituisci con il nome del tuo modello di embedding su Ollama
OLLAMA_URL = "http://ollama:11434"
QDRANT_HOST = "qdrant"
LLM_MODEL = "gemma4:e2b"
TEMPERATURE = 0.0  # temperatura per risposte più deterministiche, da finetunare
K = 6  # numero di documenti da recuperare, da finetunare
SUPPORTED_EXTENSIONS = (".py", ".json", ".txt", ".sh", ".html", ".css")
PYTHON_AST_MAX_CHARS = 1500
