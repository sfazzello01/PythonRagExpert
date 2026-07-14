# config.py

EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_URL = "http://ollama:11434"
QDRANT_HOST = "qdrant"
QDRANT_COLLECTION = "codebase_expert"

LLM_MODEL = "llama3.2:3b" 
TEMPERATURE = 0.0  # Zero tolleranza per l'inventiva, risposte deterministiche

# Numero di chunk primari da recuperare (ridotto da 25 per evitare di saturare la VRAM)
K = 4  

SUPPORTED_EXTENSIONS = (".py", ".json", ".txt", ".sh", ".html", ".css")
PYTHON_AST_MAX_CHARS = 1500

# Chunk size ridotto per evitare l'offloading in CPU e velocizzare la KV Cache
INDEX_CHUNK_SIZE = 1000  
INDEX_CHUNK_OVERLAP = 200  
INDEX_MAX_FILE_CHARS = 200_000