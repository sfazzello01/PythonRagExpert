# Esperto di Codebase

**Esperto per codebase Python** è un sistema RAG (Retrieval-Augmented Generation) che risponde in italiano a domande
su una codebase Python. Si carica lo ZIP del progetto, Esperto lo indicizza e da quel momento
risponde citando file, classi e funzioni da cui ha preso l'informazione.

Tutto gira in locale con Docker: il modello linguistico è servito da Ollama, i vettori stanno
in Qdrant e nessun dato esce dalla macchina.

## Come funziona

**Indicizzazione.** Ogni file Python viene diviso lungo i confini del codice, non a blocchi di
lunghezza fissa: un chunk per funzione o classe, con gli import del modulo come contesto. Le
classi molto lunghe diventano un'intestazione con le firme dei metodi più un chunk per metodo.
In parallelo un'analisi statica costruisce il *symbol graph*: chi chiama chi, chi eredita da
chi. I chunk sono codificati con BGE-M3 e salvati in Qdrant.

**Interrogazione.** Una domanda attraversa questi stadi:

1. **Planner** (facoltativo): il modello scompone la domanda in sotto-domande, guidato da una
   mappa del repository.
2. **Ricerca densa**: i chunk più simili a ogni domanda, con una soglia di rilevanza.
3. **Simboli nominati**: se la domanda cita `Classe.metodo`, il chunk che lo definisce entra
   direttamente.
4. **Stesso file**: altri chunk dei file già trovati.
5. **Symbol graph**: i vicini dei risultati (chiamati, chiamanti, classi base), ordinati con
   Personalized PageRank.
6. **Rerank e budget**: tutti i candidati sono rivalutati sulla domanda originale e tagliati
   alla finestra di contesto del modello.
7. **Generazione**: il modello risponde in streaming usando solo il contesto recuperato.

La descrizione completa, con formati dei dati, API e motivazioni, è in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Servizi

| Servizio | Porta | Ruolo |
|---|---|---|
| `frontend_esperto` | 8005 | Interfaccia web: chat, caricamento dello ZIP, stato dell'indice |
| `esperto` | 5000 | API di interrogazione: planner, retrieval, risposta in streaming |
| `esperto_indexer` | 5001 (solo rete interna) | Riceve lo ZIP, lo valida e lancia l'indicizzazione in background |
| `qdrant` | 6333 | Database vettoriale |
| `ollama` | 11434 | Modello linguistico locale |

## Avvio rapido

Servono Docker con Docker Compose. Una GPU NVIDIA è consigliata ma non indispensabile.

```bash
# 1. Configurazione
cp .env.example .env

# 2. Avvio dello stack
docker compose --profile esperto up -d --build

# 3. Download del modello indicato in LLM_MODEL (solo la prima volta)
docker exec ollama ollama pull qwen3.5:4b
```

Poi:

1. apri <http://localhost:8005>;
2. carica lo ZIP della codebase e attendi che lo stato dell'indice diventi `ok`;
3. scrivi la domanda.

Al primo avvio viene scaricato anche BGE-M3 (circa 2 GB), che resta in `hf_cache/`.

### Senza GPU NVIDIA

- In `docker-compose.yaml` elimina il blocco `deploy` del servizio `ollama`, segnalato dai
  commenti.
- In `.env` imposta `INDEXER_GPU=0`: l'indicizzazione usa la CPU con ONNX Runtime (e OpenVINO
  sulle CPU Intel).

## Configurazione

Tutti i parametri stanno in `.env`. I più utili:

| Variabile | Significato |
|---|---|
| `LLM_MODEL` | Modello Ollama che pianifica e risponde |
| `LLM_CONTEXT_TOKENS` | Finestra del modello; da qui deriva il budget del contesto |
| `K` | Chunk restituiti dalla ricerca densa per ogni domanda |
| `SEMANTIC_RELEVANCE_RATIO` | Soglia di rilevanza, come quota della similarità migliore |
| `SYMBOL_GRAPH_MAX_NEIGHBORS` | Vicini del grafo aggiunti per ogni domanda |
| `SYMBOL_GRAPH_MAX_DEPTH` | Profondità massima della visita del grafo |
| `INDEXER_GPU` | `1` per indicizzare su GPU NVIDIA, `0` per la CPU |
| `SUPPORTED_EXTENSIONS` | Estensioni dei file indicizzati |

L'elenco completo è nella sezione 11 di [ARCHITECTURE.md](ARCHITECTURE.md).

## Uso via API

```bash
# Stato dell'indice
curl http://localhost:5000/health

# Domanda con risposta in streaming (un evento JSON per riga)
curl -N -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"message": "Come viene validato lo ZIP caricato?"}'

# Solo retrieval, senza generazione
curl -X POST http://localhost:5000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"message": "Come viene validato lo ZIP caricato?"}'
```

Lo stream di `/ask` emette prima l'evento `sources` (fonti e sotto-domande), poi gli eventi
`answer` con il testo della risposta.

## Struttura del repository

```
esperto/                 Backend
  config.py              Configurazione letta e validata da .env
  src/api.py             API di interrogazione
  src/ingestion/         Upload dello ZIP, validazione, job di indicizzazione
  src/core/indexing/     Chunking AST, symbol graph, mappa del repository, pipeline
  src/core/rag/          Planner, retrieval, prompt, generazione
  src/core/services/     BGE-M3, Qdrant, cache degli embedding, stato dell'indice
frontend_esperto/        Interfaccia web (FastAPI, Jinja2, JavaScript)
docker-compose.yaml      Definizione dei servizi
ARCHITECTURE.md          Documentazione tecnica completa
```

Il servizio `benchmark` citato in `docker-compose.yaml` e in `ARCHITECTURE.md` non è incluso
in questo repository.

## Dati persistenti

| Cartella | Contenuto |
|---|---|
| `qdrant_data/` | Indice vettoriale |
| `esperto_state/` | Symbol graph, mappa del repository, cache degli embedding, stato dei job |
| `ollama_data/` | Modelli Ollama |
| `hf_cache/` | Modello BGE-M3 |

Ricaricare lo stesso ZIP non ricostruisce l'indice: se i file non sono cambiati
l'indicizzazione viene saltata. Quando cambiano, la cache degli embedding evita di ricodificare
i chunk rimasti uguali.

## Limiti

Esperto è un progetto di tesi pensato per l'uso in locale.

- **Nessuna autenticazione**: chiunque raggiunga le porte può interrogare il modello o avviare
  una reindicizzazione. Non va esposto su reti non fidate.
- **Solo Python**: chunking semantico e symbol graph valgono per i file `.py`; gli altri file
  supportati sono divisi in blocchi di testo.
- **Una codebase alla volta**: un nuovo ZIP sostituisce l'indice precedente.
- **Risposte da verificare**: il prompt vincola il modello al codice recuperato, ma un modello
  piccolo può comunque sbagliare. Le fonti mostrate sotto ogni risposta servono a controllare.

(il microservizio di benchmark non è attualemente caricato nella repo)
