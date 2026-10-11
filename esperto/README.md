# Esperto — RAG per codebase Python

Microservizio di question answering su codebase: indicizzazione dei vettori
densi di BGE-M3 su Qdrant, generazione in streaming via Ollama, ingestion di
archivi ZIP.

## Architettura

| Componente | Ruolo |
|---|---|
| `src/api.py` | API pubblica: `/ask` (streaming NDJSON), `/health` |
| `src/ingestion/` | Servizio indexer: upload ZIP, validazione, job di indicizzazione in background |
| `src/core/indexing/` | Discovery sorgenti, chunking AST Python, symbol graph, pipeline di upsert |
| `src/core/rag/` | Planner (sotto-domande), retrieval, prompt, chain di streaming |
| `src/core/services/` | BGE-M3 embeddings, client Qdrant, stato indice, symbol graph |
| `src/core/shared/` | Opzioni di retrieval, utility di testo |

Pipeline di retrieval: planner (sotto-domande) → ricerca densa e simboli nominati
→ chunk dello stesso file → espansione via symbol graph → rerank sulla domanda
originale → budget di contesto (max sorgenti / max caratteri).

## Test

```bash
cd esperto
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

I test sono unit test puri: nessuna dipendenza da Ollama/Qdrant
(l'ambiente minimo è impostato in `tests/conftest.py`).

## Security considerations / limitazioni note

Ambito: rete Docker interna, ambiente di sviluppo/tesi. Prima di un'esposizione
pubblica considerare:

- **Nessuna autenticazione** sugli endpoint `/ask` e `/upload`: chiunque
  raggiunga le porte può interrogare il modello o avviare re-indicizzazioni.
- **Porte esposte sull'host** (6333 Qdrant, 11434 Ollama):
  comode in sviluppo, da chiudere o vincolare in produzione.
- **Upload ZIP**: la validazione copre magic bytes, checksum, path traversal,
  estensioni e dimensione massima, ma non c'è scansione antivirus né quota
  per utente.
- **LLM locale**: le risposte sono generate da un modello senza filtri di
  output; il prompt impone grounding sul contesto ma non garantisce
  l'assenza di contenuti errati (hallucination residuali).
