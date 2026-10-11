# ARCHITECTURE.md — Sistema "Esperto di Codebase"

> Documento di architettura del sistema RAG (Retrieval-Augmented Generation) per l'interrogazione di codebase Python.
> Ultimo aggiornamento: 2026-10-09.

## Indice

1. [Panoramica](#1-panoramica)
2. [Architettura di alto livello](#2-architettura-di-alto-livello)
3. [Vista di deployment](#3-vista-di-deployment)
4. [Struttura del codice](#4-struttura-del-codice)
5. [Pipeline di indicizzazione](#5-pipeline-di-indicizzazione)
6. [Pipeline di interrogazione RAG](#6-pipeline-di-interrogazione-rag)
7. [Modello dati Qdrant](#7-modello-dati-qdrant)
8. [Symbol graph](#8-symbol-graph)
9. [API Reference](#9-api-reference)
10. [Frontend](#10-frontend)
11. [Configurazione](#11-configurazione)
12. [Persistenza e volumi](#12-persistenza-e-volumi)
13. [Sicurezza](#13-sicurezza)
14. [Testing](#14-testing)
15. [Decisioni architetturali](#15-decisioni-architetturali)
16. [Limitazioni note](#16-limitazioni-note)

---

## 1. Panoramica

**Esperto** è un assistente tecnico per codebase basato su **Retrieval-Augmented Generation**. L'utente carica l'archivio ZIP di un progetto software attraverso un'interfaccia web; il sistema lo indicizza in un database vettoriale e risponde a domande in linguaggio naturale con risposte generate da un LLM locale, **ancorate esclusivamente al codice indicizzato** e corredate dalle fonti (file, simbolo, righe).

### 1.1 Caratteristiche chiave

- **Retrieval denso**: ricerca per similarità sui vettori di BGE-M3 (1024 dimensioni).
- **Tabella dei simboli**: i nomi di classi, funzioni e metodi citati nella domanda sono agganciati in modo esatto al chunk che li definisce.
- **Mappa del repository**: README, cartelle e gerarchia delle classi, inserita in ogni prompt e passata al planner.
- **Chunking semantico AST-aware**: i file Python sono suddivisi per funzione/classe tramite parsing AST, preservando gli import di contesto; fallback a chunking testuale per gli altri formati.
- **Symbol graph statico**: grafo di chiamate/ereditarietà/import costruito via AST (senza eseguire il codice) e usato a runtime per espandere il contesto con simboli correlati, tramite BFS bidirezionale e Personalized PageRank.
- **Planner LLM**: decomposizione opzionale della domanda in sotto-domande per interrogazioni multi-parte.
- **Ingestion asincrona**: upload ZIP validato lato sicurezza, estrazione in staging, indicizzazione in job di background isolato in subprocess, nuovo indice costruito in una collection separata e attivato con uno scambio atomico di alias Qdrant.
- **Streaming NDJSON**: risposta in streaming con eventi tipizzati (`sources`, `reasoning`, `answer`).
- **Idempotenza dell'indice**: fingerprint SHA256 dei sorgenti per evitare re-indicizzazioni inutili.

### 1.2 Stack tecnologico

| Componente | Tecnologia | Versione |
|---|---|---|
| Linguaggio | Python | 3.10 |
| API framework | FastAPI + Uvicorn | 0.141.1 / 0.52.1 |
| Validazione | Pydantic | 2.13.x |
| Orchestrazione LLM | LangChain (core, ollama, text-splitters) | 1.5.3 / 1.1.0 / 1.1.2 |
| Vector database | Qdrant | v1.19.1 (server) / 1.19.0 (client) |
| LLM runtime | Ollama | 0.33.3 |
| Embedding model | BGE-M3 (BAAI/bge-m3), tokenizer `tokenizers` | 0.23.2 |
| Inferenza embedding | ONNX Runtime (CPU) con provider OpenVINO su CPU Intel, per query e indexing | 1.23.0 |
| Frontend | Jinja2 + HTML/JS (no framework) | 3.1.6 |
| Container | Docker + docker-compose | — |

---

## 2. Architettura di alto livello

Il sistema è composto da **cinque servizi** orchestrati con docker-compose, più un servizio opzionale di **benchmark** (profilo compose `benchmark`, vedi [§14](#14-testing)). I tre servizi applicativi (`esperto`, `esperto_indexer`, `frontend_esperto`) condividono la stessa base di codice Python (i primi due) e comunicano via HTTP interno.

```mermaid
graph TB
    subgraph Browser["Client"]
        U["Browser utente"]
    end

    subgraph App["Servizi applicativi"]
        FE["frontend_esperto<br/>FastAPI + Jinja2<br/>:8005"]
        BE["esperto<br/>API RAG FastAPI<br/>:5000"]
        IN["esperto_indexer<br/>Ingestion FastAPI<br/>:5001"]
    end

    subgraph Infra["Infrastruttura"]
        QD[("qdrant<br/>Vector DB<br/>:6333")]
        OL["ollama<br/>LLM runtime<br/>:11434"]
        BGE["BGE-M3<br/>modello embedding<br/>(in-process, HuggingFace)"]
    end

    U -->|"HTTP :8005"| FE
    FE -->|"POST /ask (stream)"| BE
    FE -->|"POST /upload, GET /status"| IN
    BE -->|"query_points / retrieve"| QD
    IN -->|"upsert / create_collection"| QD
    BE -->|"ChatOllama astream"| OL
    BE -.->|"encode query (CPU, lazy)"| BGE
    IN -.->|"encode documenti (CPU, GPU con INDEXER_GPU)"| BGE

    BE --- ST[("esperto_state")]
    IN --- ST
    IN --- WS[("esperto_workspace")]
    BE --- WS
```

### 2.1 Ruoli e responsabilità

| Servizio | Immagine / Build | Porta | Ruolo |
|---|---|---|---|
| `frontend_esperto` | build `./frontend_esperto` | 8005 (esposta) | UI web: chat con streaming, upload ZIP, badge di stato sistema |
| `esperto` | build `./esperto` | 5000 (esposta) | API di interrogazione: `/ask` (streaming NDJSON), `/retrieve` e `/health` |
| `esperto_indexer` | build `./esperto` (stessa immagine, comando diverso) | 5001 (solo interna) | Ingestion: `/upload`, `/status`, `/health`; esegue l'indicizzazione |
| `qdrant` | `qdrant/qdrant:v1.19.1` | 6333 | Vector database: collezione dei vettori densi |
| `ollama` | `ollama/ollama:0.33.3` | 11434 | Runtime LLM locale per la generazione in streaming |
| `benchmark` (profilo `benchmark`) | build `./benchmark` | 5002 (esposta) | Valutazione: indicizza un repository di riferimento, interroga `esperto` su un gold set e produce il report |

### 2.2 Separazione dei processi

`esperto` e `esperto_indexer` condividono codice e immagine Docker ma sono **processi separati per design**:

- L'**indexer** esegue il lavoro pesante e sincrono (parsing AST, embedding in batch, upsert) senza bloccare il servizio di query.
- Il **servizio di query** carica il modello di embedding solo per le query (lazy, su CPU con ONNX Runtime) e mantiene latenze contenute.
- Lo stato condiviso transita solo attraverso **volumi su disco** (`esperto_state`, `esperto_workspace`) e **Qdrant**: non esiste comunicazione diretta tra i due processi.

---

## 3. Vista di deployment

### 3.1 Definizione servizi (docker-compose)

```mermaid
graph LR
    subgraph Compose["docker-compose"]
        direction TB
        OL["ollama<br/>ollama/ollama:0.33.3"]
        QD["qdrant<br/>qdrant/qdrant:v1.19.1"]
        BE["esperto<br/>build ./esperto"]
        IN["esperto_indexer<br/>build ./esperto"]
        FE["frontend_esperto<br/>build ./frontend_esperto"]
    end

    BE -->|"depends_on: started"| OL
    BE -->|"depends_on: healthy"| QD
    IN -->|"depends_on: started"| OL
    IN -->|"depends_on: healthy"| QD
    FE -->|"depends_on: started"| BE
    FE -->|"depends_on: healthy"| IN
```

| Servizio | Volumi | Variabili d'ambiente chiave | Healthcheck | GPU |
|---|---|---|---|---|
| `ollama` | `./ollama_data:/root/.ollama` | `OLLAMA_CONTEXT_LENGTH=${LLM_CONTEXT_TOKENS:-20000}`, `OLLAMA_KV_CACHE_TYPE=q4_0`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1` | — | NVIDIA ×1 (opzionale) |
| `qdrant` | `./qdrant_data:/qdrant/storage` | — | TCP :6333, interval 10s, retries 12 | — |
| `esperto` | `./hf_cache:/root/.cache/huggingface`, `./esperto_state:/app/state`, `esperto_workspace:/app/workspace`, `./esperto_debug:/app/debug` | `.env` + `HF_HOME`, `PROMPT_DEBUG_FILE=/app/debug/last_prompt.json` | `curl -f http://127.0.0.1:5000/health`, start_period 30s, retries 20 | — |
| `esperto_indexer` | `./hf_cache:/root/.cache/huggingface`, `./esperto_state:/app/state`, `esperto_workspace:/app/workspace` | `.env` + `HF_HOME`, `INGESTION_WORKSPACE_DIR=/app/workspace` | `curl -f http://127.0.0.1:5001/health`, start_period 15s | NVIDIA ×1 con `INDEXER_GPU=1` |
| `frontend_esperto` | — | `ESPERTO_URL=http://esperto:5000`, `ESPERTO_INDEXER_URL=http://esperto_indexer:5001`, `PYTHONPATH=/app` | — | — |
| `benchmark` | `./benchmark/datasets:/app/datasets`, `./benchmark_results:/app/results` | `.env` + `ESPERTO_URL`, `ESPERTO_INDEXER_URL`, `OLLAMA_URL` | — | — |

Volume condiviso dichiarato in compose: `esperto_workspace` (named volume, montato da entrambi i servizi `esperto*`).

### 3.2 Dipendenze di avvio

1. `ollama` e `qdrant` partono per primi (infrastruttura).
2. `esperto` attende `ollama` avviato e `qdrant` healthy; `esperto_indexer` idem.
3. `frontend_esperto` attende `esperto` avviato e `esperto_indexer` healthy.
4. `benchmark` (solo con `--profile benchmark`) attende `esperto` ed `esperto_indexer` healthy.

### 3.3 Configurazione Ollama

Il container Ollama è configurato per un uso RAG intensivo:

- `OLLAMA_CONTEXT_LENGTH`: la finestra del modello, uguale a `LLM_CONTEXT_TOKENS` (default 20000). `esperto` passa lo stesso valore a Ollama come `num_ctx` e ne ricava il budget del contesto recuperato: due terzi della finestra, cioè 40000 caratteri stimando circa 3 caratteri per token.
- `OLLAMA_KV_CACHE_TYPE=q4_0`: KV cache quantizzata per ridurre l'occupazione di memoria GPU.
- `OLLAMA_FLASH_ATTENTION=1`: flash attention per efficienza.
- `OLLAMA_KEEP_ALIVE=-1`: il modello non viene mai scaricato dalla memoria tra una richiesta e l'altra.

---

## 4. Struttura del codice

### 4.1 Backend `esperto/`

```
esperto/
├── config.py                    # AppConfig (Pydantic): validazione centralizzata delle env
├── dockerfile                   # python:3.10-slim, dipendenze, EXPOSE 5000
├── pyproject.toml               # configurazione mypy
├── requirements.txt             # ONNX Runtime, nessun PyTorch
├── requirements-gpu.txt         # PyTorch (CUDA) e Transformers, installati solo con INDEXER_GPU=1
├── requirements-dev.txt         # requirements.txt + pytest (fuori dalle immagini)
└── src/
    ├── api.py                   # FastAPI :5000 — POST /ask (streaming), POST /retrieve, GET /health
    ├── core/
    │   ├── indexing/            # pipeline di indicizzazione
    │   │   ├── pipeline.py      #   orchestrazione: build_index(), _chunk_sources()
    │   │   ├── discovery.py     #   discovery file: iter_source_files(), relative_path(), service_for_path()
    │   │   ├── chunking.py      #   PythonASTContextualChunker + fallback_text_split()
    │   │   ├── symbol_graph.py  #   grafo statico: build_symbol_graph(), chunk_relations(), save_symbol_graph()
    │   │   └── repo_map.py      #   mappa del repository per planner e prompt: build_repo_map(), save_repo_map()
    │   ├── rag/                 # pipeline di interrogazione
    │   │   ├── chain.py         #   stream_expert_response(): composizione retriever + LLM
    │   │   ├── retrieval.py     #   Retriever.retrieve(): candidati, rerank finale, budget
    │   │   ├── planner.py       #   plan_subquestions(), parse_subquestions()
    │   │   └── prompts.py       #   build_prompt(), INSTRUCTION (regole di grounding)
    │   ├── services/            # integrazione servizi esterni
    │   │   ├── qdrant.py        #   client sync/async, create_collection(), activate_collection(), dense_search(), upsert_encoded_documents()
    │   │   ├── embedding.py     #   ciclo di vita BGE-M3: load_indexing_model(), get_query_model(), encode_*()
    │   │   ├── embedding_cache.py #  cache SQLite dei vettori dei chunk, per hash del testo codificato
    │   │   ├── index_state.py   #   fingerprint e stato: source_fingerprint(), is_current(), save()
    │   │   └── symbol_graph.py  #   SymbolGraph runtime: related_chunks() via BFS + PageRank, chunks_named()
    │   └── shared/
    │       ├── options.py       #   RetrievalOptions: interruttori per richiesta (ablation)
    │       └── text.py          #   sanitize_text(), without_imports()
    ├── ingestion/               # servizio di ingestion (:5001)
    │   ├── api.py               #   POST /upload, GET /status, GET /health
    │   ├── workspace.py         #   job in background, staging, attivazione atomica, subprocess
    │   └── zip_validation.py    #   validate_zip(): magic bytes, path traversal, integrità
    └── tests/                   # pytest: chain, chunking, embedding_cache, planner, repo_map, retrieval, retrieval_options, symbol_graph, zip_validation
```

### 4.2 Frontend `frontend_esperto/`

```
frontend_esperto/
├── config.py                    # ESPERTO_URL, ESPERTO_INDEXER_URL, directory template/static
├── dockerfile
├── requirements.txt             # fastapi, uvicorn, jinja2, requests, python-multipart
└── src/
    ├── main.py                  # route: GET /, GET /api/status, POST /api/index/upload, POST /ask
    ├── classi.py                # ChatFrontendRequest (Pydantic)
    ├── logcon.py                # logger centralizzato (logs/frontend_esperto.log)
    ├── templates/
    │   ├── base.html            #   layout: header sticky, badge stato sistema, dialog
    │   └── esperto.html         #   pagina chat: form domanda, toggle opzioni, upload, risultati
    └── static/
        ├── ui.js                #   notify(), setBusy(), confirmAction(), shortcut Ctrl+Enter
        └── style.css            #   dark theme con variabili CSS, griglia responsive
```

### 4.3 Benchmark `benchmark/`

```
benchmark/
├── config.py                    # URL dei servizi, modello della baseline closed-book, directory dataset/risultati
├── dockerfile                   # python:3.10-slim + git (clone del repository di riferimento)
├── requirements.txt             # fastapi, httpx, pyyaml, pydantic
├── datasets/
│   ├── requests/                #   psf/requests v2.32.3: 62 file, 88 domande
│   │   ├── dataset.yaml         #     repository e ref fissato
│   │   └── questions.yaml       #     gold set in 6 categorie, con simboli rilevanti e fatti chiave
│   ├── requests_en/             #   stesse domande di requests tradotte in inglese
│   └── piccolo/                 #   piccolo-orm/piccolo 1.22.0: 407 file, 60 domande (repository medio e meno noto)
├── src/
│   ├── api.py                   # FastAPI :5002 — avvio, ripresa, stato, verdetti, report e confronto dei run
│   ├── runner.py                # RunSpec, CONFIGS (ablation), campionamento, stadi retrieval e generazione
│   ├── dataset.py               # caricamento e validazione del gold set, clone e ZIP del repository
│   ├── esperto_client.py        # client HTTP verso esperto e indexer, con tempi dello stream
│   ├── metrics.py               # metriche di retrieval, stato dei simboli attesi, bootstrap
│   ├── ollama.py                # client minimale per l'API chat di Ollama (baseline closed-book)
│   └── report.py                # diagnosi e report.md, summary.json/csv, confronto tra run
└── tests/                       # pytest: api, dataset, metrics, runner
```

### 4.4 Moduli chiave e simboli principali

| Modulo | Simboli chiave | Responsabilità |
|---|---|---|
| [esperto/src/api.py](esperto/src/api.py) | `ChatRequest`, `ask_expert()`, `retrieve()`, `health_check()` | Contratto HTTP del servizio di query |
| [esperto/src/core/indexing/pipeline.py](esperto/src/core/indexing/pipeline.py) | `build_index()` | Orchestrazione end-to-end dell'indicizzazione |
| [esperto/src/core/indexing/chunking.py](esperto/src/core/indexing/chunking.py) | `PythonASTContextualChunker`, `fallback_text_split()` | Chunking semantico Python + fallback testuale |
| [esperto/src/core/indexing/discovery.py](esperto/src/core/indexing/discovery.py) | `iter_source_files()`, `relative_path()` | Scoperta dei file sorgente con filtri |
| [esperto/src/core/indexing/symbol_graph.py](esperto/src/core/indexing/symbol_graph.py) | `build_symbol_graph()`, `chunk_relations()`, `save_symbol_graph()` | Grafo statico di simboli/relazioni via AST; relazioni per chunk da scrivere nell'intestazione |
| [esperto/src/core/indexing/repo_map.py](esperto/src/core/indexing/repo_map.py) | `build_repo_map()`, `save_repo_map()` | Mappa del repository (README, cartelle, gerarchia delle classi) per planner e prompt |
| [esperto/src/core/rag/chain.py](esperto/src/core/rag/chain.py) | `stream_expert_response()` | Composizione retrieval → prompt → streaming LLM |
| [esperto/src/core/rag/retrieval.py](esperto/src/core/rag/retrieval.py) | `Retriever.retrieve()` | Candidati per domanda, rerank finale, quota del grafo, budget contesto |
| [esperto/src/core/rag/planner.py](esperto/src/core/rag/planner.py) | `plan_subquestions()`, `parse_subquestions()` | Decomposizione LLM della domanda |
| [esperto/src/core/rag/prompts.py](esperto/src/core/rag/prompts.py) | `build_prompt()`, `INSTRUCTION` | Template di prompt con regole anti-allucinazione |
| [esperto/src/core/services/qdrant.py](esperto/src/core/services/qdrant.py) | `create_collection()`, `activate_collection()`, `dense_search()`, `rerank_chunk_ids()` | Integrazione Qdrant, alias dell'indice attivo |
| [esperto/src/core/services/embedding.py](esperto/src/core/services/embedding.py) | `load_indexing_model()`, `get_query_model()` | Ciclo di vita BGE-M3 |
| [esperto/src/core/services/index_state.py](esperto/src/core/services/index_state.py) | `source_fingerprint()`, `is_current()` | Idempotenza dell'indice |
| [esperto/src/core/services/symbol_graph.py](esperto/src/core/services/symbol_graph.py) | `SymbolGraph.related_chunks()`, `SymbolGraph.chunks_named()` | Espansione grafo a runtime (BFS + PageRank); tabella dei simboli per nome |
| [esperto/src/ingestion/workspace.py](esperto/src/ingestion/workspace.py) | `start_index_job()`, `extract_to_workspace()` | Job di indicizzazione in background |
| [esperto/src/ingestion/zip_validation.py](esperto/src/ingestion/zip_validation.py) | `validate_zip()` | Validazione sicurezza archivi ZIP |
| [esperto/config.py](esperto/config.py) | `AppConfig` | Configurazione centralizzata e validata |

---

## 5. Pipeline di indicizzazione

L'indicizzazione è attivata esclusivamente dall'**upload di uno ZIP** tramite l'ingestion service. Non esiste indicizzazione di una codebase "live": la sorgente di verità è sempre l'archivio caricato.

### 5.1 Flusso end-to-end

```mermaid
flowchart TB
    A["POST /upload (ZIP multipart)"] --> B["validate_zip()<br/>magic bytes, dimensione, integrità,<br/>anti path-traversal, estensioni"]
    B -->|"valido"| C["start_index_job()<br/>crea job queued, mutex su job attivi"]
    C --> D["thread daemon: _execute_job()"]
    D --> E["extract_to_workspace()<br/>staging in workspace/job-uuid"]
    E --> G["_run_indexer_subprocess()<br/>python -m src.core.indexing.pipeline<br/>env: BASE_DIR=staging"]

    subgraph IDX["Subprocess di indicizzazione (build_index)"]
        direction TB
        H["iter_source_files()<br/>discovery: SUPPORTED_EXTENSIONS, SKIP_DIRS"] --> I["source_fingerprint()<br/>SHA256 modello + contenuti"]
        I --> J{"is_current()?"}
        J -->|"sì: skip"| Z["touch index_complete<br/>job → ready"]
        J -->|"no"| K["create_collection()<br/>nuova collection alias_timestamp"]
        K --> L["load_indexing_model()<br/>BGE-M3 (ONNX Runtime su CPU, PyTorch su GPU con INDEXER_GPU)"]
        L --> G2["build_symbol_graph()<br/>nodi e archi via AST"]
        G2 --> M["_chunk_sources()<br/>AST per .py, text splitter altrimenti"]
        M --> O["batch: encode_documents()<br/>+ _add_relation_headers()<br/>+ upsert_encoded_documents()"]
        O --> N["activate_collection()<br/>scambio atomico dell'alias"]
        N --> S["save_symbol_graph()<br/>grafo su disco"]
        S --> P["save()<br/>index_state.json + touch index_complete"]
        P --> Z
    end

    G --> H
    Z --> F["_activate()<br/>symlink atomico current → staging"]
    F --> Q["index_jobs.json aggiornato<br/>status=ready, indexing_seconds"]
    G -->|"eccezione"| R["status=error, cleanup staging"]
```

### 5.2 Validazione e staging dell'archivio

La funzione `validate_zip()` in [zip_validation.py](esperto/src/ingestion/zip_validation.py) applica quattro livelli di controllo prima di accettare l'upload:

1. **Formato**: estensione `.zip` e magic bytes `PK\x03\x04`.
2. **Dimensione**: rifiuto oltre `MAX_UPLOAD_MB` (default 200 MB).
3. **Integrità**: `ZipFile.testzip()` per rilevare archivi corrotti.
4. **Contenuto**: rifiuto di path traversal (`../`, percorsi assoluti), filtro per `SUPPORTED_EXTENSIONS`, rifiuto se non resta alcun file indicizzabile.

L'estrazione avviene in una directory di staging isolata `job-{uuid}` dentro il volume `esperto_workspace`. Solo a estrazione riuscita il symlink `current` viene aggiornato atomicamente (`os.replace`), così un eventuale fallimento non lascia mai la workspace in uno stato intermedio.

### 5.3 Isolamento in subprocess

L'indicizzazione vera e propria non gira nel processo dell'ingestion service ma in un **subprocess** (`python -m src.core.indexing.pipeline`) con override di `BASE_DIR`, che punta allo staging. Questo garantisce:

- **Isolamento della memoria**: il modello BGE-M3 in modalità indexing viene caricato e rilasciato nel subprocess, senza occupare la RAM del servizio long-running.
- **Isolamento dei fallimenti**: un crash durante l'indicizzazione non abbatte l'ingestion service; lo stato del job riflette l'errore (`status=error` con coda dei log).
- **Configurazione puntuale**: l'indicizzazione avviene sempre contro lo staging appena attivato, indipendentemente dalla configurazione del servizio.

### 5.4 Discovery e fingerprint

`iter_source_files()` in [discovery.py](esperto/src/core/indexing/discovery.py) scansiona `BASE_DIR`, applica il filtro sulle estensioni (`SUPPORTED_EXTENSIONS`) ed esclude le directory in `SKIP_DIRS` (`__pycache__`, `.git`, `node_modules`, …).

`source_fingerprint()` in [index_state.py](esperto/src/core/services/index_state.py) calcola un hash SHA256 che incorpora **sia la configurazione dell'indice** (modello, dimensione del vettore, lunghezza massima embedding, `CHUNKING_VERSION`) **sia il contenuto di ogni file**. Se il fingerprint coincide con quello persistito e la collection esiste già con il symbol graph, l'indicizzazione viene saltata (`is_current()` → `True`): caricare due volte lo stesso archivio è un'operazione a costo quasi nullo.

### 5.5 Chunking semantico

Definito in [chunking.py](esperto/src/core/indexing/chunking.py). Ogni chunk porta i metadati `filename`, `service`, `chunk_type`, `symbol_name`, `start_line`, `end_line` e `chunk_id` (deterministico, [§7.3](#73-identificatori-deterministici)).

**File Python** (`PythonASTContextualChunker`, parsing con `ast.parse()`):

| Nel sorgente | Chunk prodotti |
|---|---|
| Funzione o classe top-level | Un chunk `function` o `class`, decoratori inclusi |
| Classe oltre `CLASS_SPLIT_MIN_LINES=400` righe | Un chunk `class` di intestazione (firma, docstring, attributi e le firme di tutti i metodi, senza corpo) più un chunk `method` per metodo, con `symbol_name` nella forma `Classe.metodo` |
| Codice fuori dalle definizioni | Un chunk `global_body` |
| File con `SyntaxError` | Fallback testuale, come gli altri formati |

Il testo di un chunk di simbolo è composto da: righe di intestazione (file, tipo, nome), **import del modulo**, codice.

**Altri formati** (`fallback_text_split()`): `RecursiveCharacterTextSplitter` con `INDEX_CHUNK_SIZE=1200` caratteri e `INDEX_CHUNK_OVERLAP=150`; chunk `text_block` con nome `part_N`.

**Intestazione strutturale.** A ogni chunk di simbolo `_add_relation_headers()` in [pipeline.py](esperto/src/core/indexing/pipeline.py) antepone due righe ricavate dal symbol graph con `chunk_relations()`:

- `# EREDITA DA: ...`: le classi base; i metodi di una classe spezzata riportano quelle della loro classe;
- `# CHIAMA: ...`: al più `MAX_HEADER_CALLS=8` simboli chiamati, esclusi quelli dello stesso chunk.

Le righe sono aggiunte dopo l'encoding: fanno parte del testo salvato e letto dal modello, non di quello da cui si calcola il vettore.

### 5.6 Costruzione del symbol graph

`build_symbol_graph()` in [indexing/symbol_graph.py](esperto/src/core/indexing/symbol_graph.py) analizza i file Python in modo **statico**, senza eseguirli, in due passaggi:

1. `_DefinitionVisitor` elenca tutto ciò che è definito (moduli, classi, funzioni, metodi) e i nomi che ogni modulo importa. Serve a risolvere i riferimenti tra un modulo e l'altro.
2. `_GraphVisitor` crea i nodi, ciascuno collegato al `chunk_id` del chunk che lo contiene, e gli archi con la riga di origine: `calls` (chiamate), `inherits` (ereditarietà, anche da basi generiche come `Engine[T]`) e `imports`.

Un nome importato da un modulo che a sua volta lo importa (riesportazione, tipica dei file `__init__.py`) viene seguito fino alla definizione, per al più `MAX_REEXPORTS=5` passaggi.

Il grafo è costruito prima del chunking, perché le sue relazioni finiscono nell'intestazione dei chunk ([§5.5](#55-chunking-semantico)). È salvato in `esperto_state/symbol_graph.json` dopo l'attivazione del nuovo indice, con un campo `version` (`SYMBOL_GRAPH_VERSION`): un grafo di versione diversa viene ignorato.

### 5.7 Encoding e upsert

1. I chunk sono ordinati per lunghezza: ogni batch (`BATCH_SIZE`) contiene testi simili e richiede poco padding.
2. `encode_documents()` prende dalla cache `embedding_cache.sqlite` i vettori dei chunk il cui testo è già stato codificato e invoca BGE-M3 solo sugli altri.
3. `upsert_encoded_documents()` carica i punti sulla collection, con ID deterministico ([§7.3](#73-identificatori-deterministici)). Se la richiesta supera la dimensione ammessa, i punti sono inseriti uno alla volta.
4. Al termine `save()` scrive `index_state.json` (collection, fingerprint, numero di file, versione del grafo) e crea il flag `index_complete`, che sblocca l'health check del servizio di query.

---

## 6. Pipeline di interrogazione RAG

L'interrogazione è esposta come `POST /ask` sul servizio `esperto` (:5000) e produce uno **stream NDJSON**: ogni riga è un oggetto JSON indipendente con campo `type` (`sources`, `reasoning`, `answer`) e campo `data`.

### 6.1 Diagramma di sequenza

```mermaid
sequenceDiagram
    autonumber
    participant U as Browser
    participant FE as frontend_esperto :8005
    participant BE as esperto :5000
    participant PL as Planner LLM (Ollama)
    participant EM as BGE-M3 (query, CPU)
    participant QD as Qdrant :6333
    participant SG as SymbolGraph (symbol_graph.json)
    participant LL as Ollama LLM :11434

    U->>FE: POST /ask {message, use_planner, use_reasoning}
    FE->>BE: proxy POST /ask (stream)
    BE->>BE: validazione ChatRequest (lunghezza, non vuoto)
    alt use_planner = true
        BE->>PL: plan_subquestions(domanda, mappa del repository) — temperature 0
        PL-->>BE: JSON {"subquestions": [...]} (max 4, fallback: domanda originale)
    end
    BE->>EM: encode_queries([domanda] + sotto-domande)
    EM-->>BE: un vettore denso per domanda
    loop ogni domanda (originale + sotto-domande), in parallelo
        BE->>QD: dense_search: top K sopra la soglia di similarità
        QD-->>BE: candidati con _score (coseno)
        BE->>SG: chunks_named(domanda) — tabella dei simboli
        SG-->>BE: chunk dei simboli nominati
        BE->>QD: get_documents_by_chunk_ids(nominati)
        QD-->>BE: simboli nominati (in testa alle fonti)
        BE->>QD: dense_search filtrata per file (enrichment same-file)
        QD-->>BE: fino a 4 chunk per file
        BE->>SG: related_chunks(semi) — BFS + Personalized PageRank
        SG-->>BE: chunk correlati + relazioni
        BE->>QD: get_documents_by_chunk_ids(correlati)
        QD-->>BE: vicini del grafo (_graph_relation)
    end
    BE->>QD: rerank_chunk_ids: risultati diretti vs domanda originale
    QD-->>BE: similarità con la domanda originale
    BE->>BE: _apply_context_budget (quota del grafo + risultati diretti)
    BE-->>FE: evento NDJSON "sources" (fonti + sotto-domande + mappa)
    FE-->>U: rendering elenco fonti
    BE->>BE: build_prompt(mappa, contesto, domanda) + salvataggio debug
    BE->>LL: astream(prompt) — ChatOllama
    loop streaming
        LL-->>BE: chunk (answer / reasoning_content)
        BE-->>FE: eventi NDJSON "answer" / "reasoning"
        FE-->>U: rendering incrementale (markdown)
    end
```

### 6.2 Planner delle sotto-domande

Con `use_planner=true` (default nel frontend) `plan_subquestions()` in [planner.py](esperto/src/core/rag/planner.py) interroga l'LLM a temperatura 0:

- una domanda già specifica resta da sola; le altre sono scomposte in 2–4 sotto-domande autonome;
- il planner riceve la mappa del repository ([§6.5](#65-prompt-e-grounding)) e deve nominare nelle sotto-domande le classi pertinenti della gerarchia, con il nome esatto, così la tabella dei simboli le aggancia; ripiega sulle cartelle solo se nessuna classe è pertinente;
- non può introdurre nomi di file, simboli o tecnologie assenti dalla domanda e dalla mappa;
- le sotto-domande restano nella lingua della domanda.

`parse_subquestions()` valida il JSON prodotto, toglie duplicati e frammenti troppo corti e tiene al più 4 sotto-domande. Se l'output non è valido si usa la sola domanda originale.

Al retrieval partecipano la domanda originale e le sotto-domande, codificate in un solo passaggio BGE-M3 e cercate in parallelo.

### 6.3 Ricerca densa

Implementata da `Retriever` in [retrieval.py](esperto/src/core/rag/retrieval.py) e `dense_search()` in [services/qdrant.py](esperto/src/core/services/qdrant.py):

1. **Ricerca per similarità**: i primi `K` chunk per similarità coseno tra il vettore della domanda e i vettori densi dei chunk (HNSW sul vettore `dense`). La similarità è esposta come `_score` e ordina i risultati.
2. **Soglia di rilevanza**: si scartano i chunk la cui similarità è sotto `SEMANTIC_RELEVANCE_RATIO` volte la migliore.

I termini esatti del codice (nomi di classi, funzioni, metodi) non passano dalla ricerca vettoriale: li aggancia la tabella dei simboli ([§6.4](#64-costruzione-del-contesto)).

### 6.4 Costruzione del contesto

`Retriever.retrieve()` in [retrieval.py](esperto/src/core/rag/retrieval.py) lavora in due tempi.

**Per ogni domanda** (l'originale e ogni sotto-domanda), in parallelo, i candidati arrivano da quattro stadi. Ogni fonte riporta in `origin` lo stadio che l'ha trovata; se sono più di uno, il primo nell'ordine della tabella.

| Stadio | `origin` | Cosa aggiunge | Si spegne con |
|---|---|---|---|
| Ricerca densa | `search` | I primi `K` chunk sopra la soglia di rilevanza ([§6.3](#63-ricerca-densa)) | — |
| Simboli nominati | `symbol` | I chunk che definiscono i simboli citati nella domanda | `use_symbol_lookup` |
| Stesso file | `same_file` | Fino a 4 chunk per ogni file trovato dalla ricerca, i più simili alla domanda | `use_same_file` |
| Symbol graph | `graph` | Fino a `SYMBOL_GRAPH_MAX_NEIGHBORS=6` chunk collegati ai semi da chiamate o ereditarietà ([§8](#8-symbol-graph)) | `use_symbol_graph` |

- **Simboli nominati**: le parole della domanda con la forma di un identificatore (un underscore, un punto o una maiuscola) sono cercate nella tabella dei simboli (`SymbolGraph.chunks_named()`). Un nome definito in più di `MAX_NAME_MATCHES=3` chunk è ambiguo e viene ignorato.
- **Semi del grafo**: sono i risultati della ricerca e i simboli nominati, non i chunk aggiunti per stesso file. I vicini diretti dei primi `TOP_SEEDS=3` semi hanno la precedenza.

**Su tutti i candidati insieme:**

1. **Unione** per `chunk_id`.
2. **Rerank**: i candidati diretti (ricerca, simboli nominati, stesso file) sono valutati sulla domanda originale con `rerank_chunk_ids()`, perché gli score di ricerche diverse non sono confrontabili. I simboli nominati restano comunque in testa.
3. **Budget** (`_apply_context_budget()`): le fonti entrano in ordine fino a `RAG_CONTEXT_MAX_SOURCES` (default 35) e al budget di caratteri derivato da `LLM_CONTEXT_TOKENS`. Gli import del modulo contano una volta per file, come nel contesto mostrato al modello.

I vicini del grafo servono proprio quando non somigliano alla domanda, quindi saltano il rerank e hanno una quota riservata: fino a `SYMBOL_GRAPH_MAX_NEIGHBORS` posti per ogni domanda, con la quota proporzionale di caratteri e mai oltre metà del budget. Nelle fonti compaiono in coda, senza score.

### 6.5 Prompt e grounding

**Mappa del repository.** Prima del contesto recuperato il prompt include una mappa compatta e deterministica, costruita in indicizzazione da `build_repo_map()` in [repo_map.py](esperto/src/core/indexing/repo_map.py) e salvata in `esperto_state/repo_map.txt`. Ha tre parti, entro `REPO_MAP_MAX_CHARS` (il 15% del budget del contesto):

1. **README**: l'inizio del README più vicino alla radice, saltate le righe iniziali di badge e HTML (al più un quarto del budget).
2. **Cartelle**: una riga per cartella con il numero di file indicizzati, alla profondità massima che entra in metà del budget.
3. **Gerarchia delle classi**: per ogni classe base definita fuori dai test, le sottoclassi dirette (al più `MAX_SUBCLASSES=10`), ricavate dagli archi `inherits` del symbol graph; prima le classi con più sottoclassi, finché c'è spazio.

La mappa serve alle domande generali e di architettura, per cui la ricerca per somiglianza non basta. È passata anche al planner ([§6.2](#62-planner-delle-sotto-domande)) ed è inviata al client nell'evento `sources`. Si disattiva per richiesta con `use_repo_map=false`.

**Ordine del contesto.** `_build_context()` in [chain.py](esperto/src/core/rag/chain.py) raggruppa le fonti per file: i file nell'ordine del loro chunk meglio classificato, i chunk dello stesso file per riga. Il blocco degli import del modulo, presente in ogni chunk, resta solo sul primo chunk di ogni file (`without_imports()`). L'elenco `sources` inviato al client resta in ordine di rilevanza e con il testo completo.

**Prompt.** `build_prompt()` in [prompts.py](esperto/src/core/rag/prompts.py) compone un unico `HumanMessage`, perché alcuni modelli Ollama gestiscono male un `SystemMessage` separato. L'ordine è:

1. mappa del repository: è uguale a ogni richiesta, quindi in testa permette a Ollama di riusarne la cache;
2. domanda: prima del contesto, così il modello lo legge sapendo cosa cercare;
3. contesto recuperato;
4. regole;
5. di nuovo la domanda.

Le regole sono cinque:

- usare solo il codice e le informazioni del contesto, senza inventare file, funzioni o librerie;
- se la risposta non è nel contesto, dichiararlo esplicitamente;
- citare file e funzione o classe di riferimento;
- per le modifiche proposte, mostrare "Prima" e "Dopo";
- essere concisi.

Il prompt renderizzato di ogni richiesta è salvato in `esperto_debug/last_prompt.json` (`PROMPT_DEBUG_FILE`) per ispezione.

### 6.6 Streaming e gestione errori

`stream_expert_response()` in [chain.py](esperto/src/core/rag/chain.py) è un async generator che emette nell'ordine:

| `type` | `data` | Quando |
|---|---|---|
| `sources` | `{sources: [...], subquestions: [...], repo_map: "..."}` | Una volta, prima della generazione |
| `reasoning` | frammento di reasoning | Solo se `use_reasoning=true` e il modello lo produce |
| `answer` | frammento di risposta | Streaming incrementale del completamento |

Le fonti includono `filename`, `service`, `chunk_type`, `symbol_name`, `start_line`, `end_line`, `score`, `origin` (lo stadio che ha trovato il chunk, [§6.4](#64-costruzione-del-contesto)) ed eventualmente `graph_relation`. Gli errori a metà stream (durata complessiva oltre `LLM_INFERENCE_TIMEOUT`, timeout o errori di Qdrant, eccezioni generiche) sono convertiti in eventi `[ERROR] ...` invece di troncare silenziosamente la risposta.

---

## 7. Modello dati Qdrant

### 7.1 Vettori della collection

`QDRANT_COLLECTION` (default d'uso `codebase_expert_bgem3_v1`) non è una collection ma un **alias**:

- ogni indicizzazione crea una collection nuova, `<alias>_<timestamp>`;
- a costruzione finita `activate_collection()` sposta l'alias sulla nuova in un'unica operazione ed elimina la precedente;
- durante la ricostruzione, o se questa fallisce, le query continuano a vedere l'indice precedente con il suo symbol graph.

Ogni punto ha un solo vettore; la collection è creata da `create_collection()` in [services/qdrant.py](esperto/src/core/services/qdrant.py):

| Vettore | Dimensione | Distanza | Indicizzazione | Storage | Quantizzazione | Ruolo |
|---|---|---|---|---|---|---|
| `dense` | 1024 | COSINE | HNSW | RAM | INT8 scalare (quantile 0.99, sempre in RAM) | Similarità semantica a livello di chunk |

### 7.2 Payload dei punti

Ogni punto rappresenta un chunk e porta:

```json
{
  "page_content": "# FILE: src/api.py | SERVIZIO: src\n# TIPO: FUNCTION | NOME: ask_expert\n...imports e codice...",
  "metadata": {
    "filename": "src/api.py",
    "service": "src",
    "chunk_type": "function",
    "symbol_name": "ask_expert",
    "start_line": 82,
    "end_line": 106,
    "chunk_id": "9f2c…(md5)"
  }
}
```

Valori di `chunk_type`: `function`, `class`, `method`, `global_body`, `text_block`. Il campo `service` è derivato dal percorso (`service_for_path()`) e usato nei metadati esposti al client.

### 7.3 Identificatori deterministici

- **`chunk_id`**: deterministico — `MD5(filename::symbol_name::riga_def-1::end_line)` per simboli Python, `MD5(filename::part_N)` per blocchi testuali.
- **Point ID Qdrant**: `uuid5(NAMESPACE_URL, "esperto:{chunk_id}")` — UUID valido derivato deterministicamente dal chunk_id.

Questa catena rende ogni re-indicizzazione **sovrascrittura idempotente** degli stessi punti: nessun duplicato orfano nella collection, nessuna dipendenza da stato esterno per ricostruire gli ID.

---

## 8. Symbol graph

Il symbol graph è l'elemento che distingue il retrieval di Esperto da una semplice ricerca vettoriale: fornisce al sistema la **struttura del codice** (chi chiama chi, chi eredita da chi) in forma interrogabile.

### 8.1 Formato persistito

`esperto_state/symbol_graph.json`:

```json
{
  "version": 1,
  "nodes": [
    {
      "id": "src.api::ask_expert",
      "kind": "function",
      "name": "ask_expert",
      "filename": "src/api.py",
      "start_line": 82,
      "end_line": 106,
      "chunk_id": "9f2c…"
    }
  ],
  "edges": [
    {
      "source": "src.api::ask_expert",
      "target": "src.rag::stream_expert_response",
      "kind": "calls",
      "line": 95
    }
  ]
}
```

Tipi di nodo: `module`, `class`, `function`, `method`. Tipi di arco: `calls`, `inherits`, `imports`. Il campo `version` è verificato a caricamento: un grafo con versione non corrente viene ignorato (comportamento degrado graceful a retrieval senza espansione).

### 8.2 Uso a runtime

La classe `SymbolGraph` in [services/symbol_graph.py](esperto/src/core/services/symbol_graph.py) è caricata lazy dal retriever e si auto-ricarica se il file cambia su disco (`_reload_if_needed()`), così una nuova indicizzazione è recepita senza riavvio del servizio.

`chunks_named(text)` è la tabella dei simboli: al caricamento ogni nodo è indicizzato per nome corto (`login`) e qualificato (`BaseUser.login`) verso il chunk che lo contiene; la ricerca è esatta e distingue maiuscole e minuscole ([§6.4](#64-costruzione-del-contesto)).

`related_chunks(chunk_ids, limit, top_chunk_ids)` sceglie i vicini dei semi in quattro passi:

1. **Raccolta** (BFS): dai simboli dei semi si seguono gli archi `calls` e `inherits` in entrambe le direzioni, fino a profondità `SYMBOL_GRAPH_MAX_DEPTH=3`. I semi e i chunk già nel contesto sono esclusi.
2. **Punteggio** (Personalized PageRank): i candidati sono ordinati per vicinanza ai semi lungo gli archi diretti (damping `0.4`, al più 20 iterazioni, tolleranza `1e-6`). A parità di punteggio, come accade per i chiamanti, vince il percorso più corto.
3. **Precedenza ai vicini diretti** dei primi risultati (`top_chunk_ids`): i chunk a un passo da quei semi vengono prima di tutti gli altri, e tra loro prima quelli nella stessa classe o nello stesso modulo. Completano le catene di chiamate che la ricerca per somiglianza non trova.
4. **Selezione**: i primi `limit` chunk, ciascuno con il percorso più breve dai semi come relazione (esposta al client in `graph_relation`).

L'effetto pratico: se l'utente chiede di una funzione, il contesto include automaticamente anche i suoi chiamanti e le funzioni che invoca, anche se lessicalmente dissimili dalla domanda.

---

## 9. API Reference

### 9.1 Servizio di query — `esperto` (:5000)

#### `POST /ask`

Interroga il sistema con risposta in streaming.

**Request body** (`ChatRequest`):

| Campo | Tipo | Default | Vincoli |
|---|---|---|---|
| `message` | string | — | `MIN_MESSAGE_LENGTH` (3) ≤ len ≤ `MAX_MESSAGE_LENGTH` (5000), non vuoto dopo strip |
| `use_planner` | bool | `true` | attiva la decomposizione in sotto-domande |
| `use_reasoning` | bool | `false` | inoltra i frammenti di reasoning del modello, se prodotti |
| `retrieval` | object | tutti `true` | `RetrievalOptions`: `use_same_file`, `use_symbol_graph`, `use_symbol_lookup`, `use_repo_map`, più `k` (1–50, default: `K` di configurazione) |

**Response**: `StreamingResponse`, `text/plain; charset=utf-8`, una riga JSON per evento:

```json
{"type": "sources", "data": {"sources": [{"filename": "…", "symbol_name": "…", "chunk_type": "…", "score": 0.83, "origin": "search", "graph_relation": null}], "subquestions": ["…"], "repo_map": "…"}}
{"type": "reasoning", "data": "…frammento di ragionamento…"}
{"type": "answer", "data": "…frammento di risposta…"}
```

**Errori**: `400` messaggio non valido; `500` errore interno; eventi `[ERROR]` inline per timeout/errori a stream avviato.

#### `POST /retrieve`

Planner + retrieval senza generazione, usato dal benchmark. Stesso corpo di `/ask` senza `use_reasoning`, più `subquestions` opzionale: un piano già calcolato che salta il planner; risponde con `{"subquestions": [...], "sources": [...], "dropped": [...]}`, dove `dropped` elenca senza testo i candidati rimasti fuori dal contesto: permette di distinguere un chunk mai trovato da uno trovato e tagliato dal budget. Un errore o timeout di Qdrant produce `500`, mai una lista di fonti vuota.

#### `GET /health`

Stato dell'indice, usato da healthcheck Docker e dal badge del frontend:

| Stato | Significato |
|---|---|
| `not_indexed` | flag `index_complete` assente: caricare uno ZIP |
| `indexing` | flag presente ma collection vuota: indicizzazione in corso |
| `ok` | indice pronto; include `points_count` |

### 9.2 Ingestion service — `esperto_indexer` (:5001, solo rete interna)

#### `POST /upload`

Accetta un archivio ZIP multipart (campo `file`), lo valida e accoda il job di indicizzazione.

- **Successo**: `HTTP 202` con `{status: "queued", job: {job_id, filename, status, files, started_at}}`.
- **Errori**: `400` archivio non valido (estensione, magic bytes, integrità, path traversal, nessun file supportato, oltre `MAX_UPLOAD_MB`); `409` un job è già `queued` o `indexing`.

#### `GET /status`

Stato del job corrente: `{status: "not_indexed"|"queued"|"indexing"|"ready"|"error", job: {...}}` con `files`, `started_at`, `finished_at`, `error`, `indexing_seconds`. Job rimasti `queued`/`indexing` dopo un riavvio sono marcati `error` al caricamento (job stale).

#### `GET /health`

`{"status": "ok"}` — usato dal healthcheck Docker.

### 9.3 Frontend — `frontend_esperto` (:8005)

| Metodo | Path | Funzione |
|---|---|---|
| GET | `/` | Pagina chat (`esperto.html`) con badge di stato sistema |
| GET | `/api/status` | Aggrega `/health` di `esperto` e `/status` dell'indexer |
| POST | `/api/index/upload` | Proxy multipart verso indexer `/upload` (timeout 10s/120s) |
| POST | `/ask` | Proxy streaming verso `esperto` `/ask` (timeout 10s/300s) |

Il frontend agisce da **proxy trasparente**: nessuna logica RAG risiede qui; valida solo la forma della richiesta (`ChatFrontendRequest`) e inoltra lo stream.

### 9.4 Benchmark — `benchmark` (:5002)

| Metodo | Path | Funzione |
|---|---|---|
| POST | `/runs` | Avvia un run in background (`202`, restituisce `run_id`); `409` se un run è già in corso |
| POST | `/runs/{run_id}/resume` | Riprende un run interrotto con la stessa specifica, saltando indicizzazione e domande già completate |
| GET | `/runs` | Elenco dei run presenti in `benchmark_results/` |
| GET | `/runs/{run_id}` | Avanzamento (`status.json`: stadio, completate, totale, errore) |
| GET | `/runs/{run_id}/report` | Report Markdown del run concluso |
| POST | `/runs/{run_id}/judgments` | Riceve i verdetti sulle risposte di `to_judge.jsonl` e rigenera il report |
| GET | `/runs/{run_id}/compare/{other_id}` | Differenze appaiate tra due run (questo − l'altro) sulle configurazioni e domande in comune |
| POST | `/datasets/{nome}/validate` | Verifica che file e simboli del gold set esistano nel repository |

Corpo di `POST /runs` (`RunSpec`):

| Campo | Default | Descrizione |
|---|---|---|
| `dataset` | — | Nome della cartella in `benchmark/datasets/` |
| `stages` | `["retrieval", "generation"]` | Stadi da eseguire |
| `retrieval_configs` | tutte | Configurazioni su cui misurare il retrieval (vedi sotto) |
| `generation_configs` | `["full", "naive", "naive_budget", "closed_book"]` | Configurazioni per cui generare le risposte |
| `per_category` | `4` | Domande estratte per categoria; `null` = tutto il gold set |
| `seed` | `0` | Seme del campionamento |
| `skip_index` | `false` | Riusa l'indice attivo; rifiutato se non è quello del dataset |
| `planner` | `true` | `false` spegne il planner in tutte le configurazioni |

Configurazioni: ognuna differisce da `full` per un solo elemento. Il suffisso `@k=N` fissa il K di qualunque configurazione (es. `full@k=20`), per provare più valori senza riavviare `esperto`.

| Nome | Rispetto a `full` |
|---|---|
| `full` | Sistema completo |
| `no_planner` | Senza planner |
| `no_symbol_graph` | Senza vicini del grafo |
| `no_same_file` | Senza chunk dello stesso file |
| `no_symbol_lookup` | Senza simboli nominati |
| `no_repo_map` | Senza mappa, nel planner e nel prompt |
| `naive` | Sola ricerca densa, senza planner né altri stadi |
| `naive_budget` | Come `naive` con `k=50`: è il budget di contesto a tagliare, come in `full` |
| `closed_book` | Solo generazione: il modello senza contesto |

Ogni run scrive in `benchmark_results/<run_id>/` (l'orario nel nome è UTC): `status.json`, `manifest.json` (specifica, commit, parametri RAG), `retrieval.jsonl` e `generation.jsonl` (una riga per domanda e configurazione), `to_judge.jsonl`, `contexts/`, `summary.json`, `summary.csv` e `report.md`.

**Categorie del gold set.** `lookup`, `behavior`, `multihop` e `architecture` hanno simboli rilevanti e fatti chiave; `overview` (domande di panoramica come "descrivi la codebase") ha solo fatti chiave, quindi conta per la generazione e non per le metriche di retrieval; `unanswerable` non ha gold e misura le risposte inventate. La cartella `requests_en` contiene le stesse domande in inglese.

**Giudizio delle risposte.** Le risposte sono giudicate fuori dal servizio, alla cieca:

- a fine generazione il servizio scrive `to_judge.jsonl`: per ogni risposta una chiave opaca, la domanda, i fatti chiave e la risposta, senza la configurazione;
- il contesto usato per ciascuna risposta è in `contexts/<chiave>.txt`;
- i verdetti (`fact_verdicts`, uno per fatto chiave, e `abstains`) si consegnano con `POST /runs/{run_id}/judgments`, che li salva in `judgments.jsonl` e rigenera il report;
- un verdetto con un numero di fatti sbagliato viene scartato e la risposta resta non giudicata.

**Metriche.** Di retrieval, sulle domande con gold: `symbol_mrr` (quanto in alto sta il primo simbolo atteso), `symbol_context_recall` (quota dei simboli attesi nel contesto) e `hop2_symbol_recall` (lo stesso sui simboli raggiungibili solo seguendo una relazione), più numero di fonti, caratteri e secondi. Di generazione, dopo il giudizio: correttezza (quota dei fatti chiave presenti), risposte inventate sulle domande senza risposta, rifiuti errati, più tempi e lunghezza della risposta.

**Report.** `report.md` è costruito per dire dove intervenire:

- **Diagnosi**: poche frasi ricavate dai dati, in testa al report.
- **Dove finiscono i simboli attesi**: per ogni configurazione, la quota di simboli attesi nel contesto divisa per stadio che li ha portati (`origin`), quelli trovati ma esclusi dal budget (`dropped`) e quelli mai trovati. Distingue un problema di ricerca da uno di budget.
- **Effetto di ogni configurazione**: differenze appaiate rispetto a `full`, con intervallo di confidenza; la diagnosi ne ricava gli stadi con e senza effetto misurabile.
- **Domande con simboli mancanti**: per `full`, quali simboli mancano in quali domande.
- **Retrieval o modello?**: correttezza delle domande con contesto completo (tutti i simboli attesi presenti) e incompleto; separa gli errori del retrieval da quelli del modello.
- **Fatti sbagliati**: ogni fatto chiave mancato, con le configurazioni che lo sbagliano e un segno quando il contesto era incompleto.

---

## 10. Frontend

### 10.1 Flussi utente

**Chat**: l'utente scrive la domanda, opzionalmente attiva/disattiva planner e reasoning, invia (anche con Ctrl+Enter). Il browser consuma lo stream NDJSON con `ReadableStream`, aggiorna incrementalmente la risposta (rendering markdown via `marked()`) e popola la colonna laterale con stato, sotto-domande generate dal planner ed elenco delle fonti con score e relazioni di grafo. Durante lo streaming è disponibile il pulsante "Interrompi" (abort della fetch).

**Upload**: selezione di uno ZIP, invio a `/api/index/upload`, quindi polling di `/api/status` per seguire l'avanzamento del job fino a `ready` (badge "Indice pronto") o `error` (con dettaglio).

### 10.2 UI e asset

- `templates/base.html`: layout comune con header sticky, badge di stato Esperto, dialog di conferma riutilizzabile.
- `templates/esperto.html`: form domanda con toggle planner/reasoning, card di upload, griglia risultati a due colonne (risposta | info e fonti).
- `static/ui.js`: utility condivise — `notify()` per avvisi dismissibili, `setBusy()` per stati di caricamento dei pulsanti, `confirmAction()` per dialog con Promise, shortcut da tastiera.
- `static/style.css`: dark theme con variabili CSS (accento `#8b2032`), layout responsive con `clamp()`.
- `logcon.py`: logging centralizzato su `logs/frontend_esperto.log`.

---

## 11. Configurazione

Tutta la configurazione del backend è centralizzata in `AppConfig` ([esperto/config.py](esperto/config.py)), un modello Pydantic che **valida all'avvio** tipo, range e presenza delle variabili d'ambiente: un errore di configurazione blocca il boot con un messaggio esplicito invece di emergere a runtime.

### 11.1 LLM (Ollama)

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `OLLAMA_URL` | obbligatoria | URL valido | URL del servizio Ollama (es. `http://ollama:11434`) |
| `LLM_MODEL` | obbligatoria | — | Modello Ollama per la generazione |
| `TEMPERATURE` | obbligatoria | 0.0–2.0 | Temperatura della risposta (il planner usa sempre 0.0) |
| `LLM_INFERENCE_TIMEOUT` | obbligatoria | ≥ 10s | Timeout dello stream di generazione |

### 11.2 Qdrant

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `QDRANT_HOST` | obbligatoria | — | Host Qdrant |
| `QDRANT_PORT` | obbligatoria | — | Porta Qdrant (6333) |
| `QDRANT_COLLECTION` | obbligatoria | — | Nome collection (es. `codebase_expert_bgem3_v1`) |
| `QDRANT_QUERY_TIMEOUT` | obbligatoria | ≥ 5s | Timeout delle query |

### 11.3 Retrieval

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `K` | obbligatoria | 1–50 | Risultati della ricerca densa per ogni domanda (originale e sotto-domande) |
| `SEMANTIC_RELEVANCE_RATIO` | obbligatoria | 0.0–1.0 | Soglia di rilevanza relativa alla similarità migliore, applicata solo alla ricerca di ogni domanda (non ai chunk dello stesso file né al rerank finale) |
| `LLM_CONTEXT_TOKENS` | 20000 | ≥ 2000 | Finestra del modello: `num_ctx` di Ollama, da cui deriva il budget del contesto (due terzi, ~3 caratteri per token) |
| `RAG_CONTEXT_MAX_SOURCES` | 35 | ≥ 1 | Numero massimo di fonti nel contesto |
| `MIN_MESSAGE_LENGTH` / `MAX_MESSAGE_LENGTH` | obbligatorie | 1–100 / 100–50000 | Limiti lunghezza domanda (valori tipici 3 / 5000) |

### 11.4 Embedding (BGE-M3)

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `DENSE_MODEL_NAME` | `BAAI/bge-m3` | — | Identificativo HuggingFace del modello |
| `INDEXER_GPU` | `0` | `0` / `1` | `1`: l'indexer codifica i chunk su GPU NVIDIA (PyTorch, fp16). Letta anche da compose per costruire l'immagine con PyTorch e assegnare la GPU: dopo averla cambiata serve `docker compose build esperto_indexer` |
| `EMBEDDING_INT8` | `false` | bool | Modello quantizzato a 8 bit sul provider CPU standard (creato al primo avvio); i vettori cambiano, quindi richiede una reindicizzazione |
| `EMBEDDING_MAX_LENGTH` | obbligatoria | 128–8192 | Lunghezza massima in token per encoding (tipicamente 1024) |
| `BATCH_SIZE` | obbligatoria | 1–128 | Batch size dell'encoding in indicizzazione (tipicamente 8) |

### 11.5 Indicizzazione e discovery

| Variabile | Default | Descrizione |
|---|---|---|
| `BASE_DIR` | obbligatoria | Radice dei sorgenti da indicizzare (nel compose: `/app/workspace/current`) |
| `SUPPORTED_EXTENSIONS` | obbligatoria | Estensioni indicizzabili (tipiche: `.py,.yaml,.yml,.md,.txt,.html,.json,.js`) |
| `SKIP_DIRS` | obbligatoria | Directory escluse (tipiche: `__pycache__,.git,node_modules,…`) |
| `INDEX_CHUNK_SIZE` / `INDEX_CHUNK_OVERLAP` | 1200 / 150 | Dimensione e overlap del fallback testuale |
| `INDEX_STATE_DIR` | obbligatoria | Directory dei file di stato (nel compose: `/app/state`) |

### 11.6 Symbol graph

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `SYMBOL_GRAPH_MAX_NEIGHBORS` | 6 | ≥ 1 | Chunk correlati massimi per domanda (quota riservata nel contesto) |
| `SYMBOL_GRAPH_MAX_DEPTH` | 3 | ≥ 0 | Profondità BFS |
| `SYMBOL_GRAPH_PAGERANK_DAMPING` | 0.4 | (0.0, 1.0) | Damping factor PageRank |
| `SYMBOL_GRAPH_PAGERANK_ITERATIONS` | 20 | ≥ 1 | Iterazioni PageRank |
| `SYMBOL_GRAPH_PAGERANK_TOLERANCE` | 1e-6 | > 0 | Tolleranza di convergenza |

### 11.7 Ingestion e debug

| Variabile | Default | Descrizione |
|---|---|---|
| `MAX_UPLOAD_MB` | 200 | Dimensione massima dello ZIP |
| `MAX_EXTRACTED_MB` | 1000 | Dimensione massima del contenuto decompresso |
| `INGESTION_WORKSPACE_DIR` | `/app/workspace` | Directory di staging degli upload |
| `PROMPT_DEBUG_FILE` | `/app/debug/last_prompt.json` | Salvataggio dell'ultimo prompt renderizzato |

### 11.8 Benchmark

Il servizio di benchmark legge lo stesso `.env` (per registrare i parametri RAG nel manifest) più le proprie variabili ([benchmark/config.py](benchmark/config.py)):

| Variabile | Default | Descrizione |
|---|---|---|
| `BENCHMARK_REQUEST_TIMEOUT` | 300 | Timeout in secondi delle richieste a esperto e Ollama |
| `BENCHMARK_INDEX_TIMEOUT` | 3600 | Attesa massima in secondi dell'indicizzazione |

---

## 12. Persistenza e volumi

### 12.1 Mapping dei volumi

| Percorso host | Percorso container | Servizi | Contenuto |
|---|---|---|---|
| `./hf_cache` | `/root/.cache/huggingface` | esperto, esperto_indexer | Cache HuggingFace del modello BGE-M3 |
| `./ollama_data` | `/root/.ollama` | ollama | Modelli LLM scaricati (blob quantizzati, manifest) |
| `./qdrant_data` | `/qdrant/storage` | qdrant | Collection vettoriale e stato RAFT |
| `./esperto_state` | `/app/state` | esperto, esperto_indexer | Stato dell'indice e dei job (vedi §12.2) |
| `./esperto_debug` | `/app/debug` | esperto | `last_prompt.json` per ispezione prompt |
| `esperto_workspace` (named) | `/app/workspace` | esperto, esperto_indexer | Staging degli upload e workspace corrente |
| `./benchmark/datasets` | `/app/datasets` | benchmark | Gold set e riferimento al repository |
| `./benchmark_results` | `/app/results` | benchmark | Risultati dei run e cache dei repository clonati (`.repos/`) |

### 12.2 File di stato (`esperto_state/`)

| File | Contenuto | Scopo |
|---|---|---|
| `index_state.json` | `{collection, fingerprint, source_file_count, symbol_graph_version}` | Idempotenza: confronto fingerprint per skip del rebuild |
| `index_complete` | file flag vuoto | Segnala indice utilizzabile; letto da `/health` |
| `index_jobs.json` | `{job_id, filename, status, files, started_at, finished_at, error, indexing_seconds}` | Stato dell'ultimo job di upload (`queued`/`indexing`/`ready`/`error`) |
| `symbol_graph.json` | `{version, nodes[], edges[]}` | Grafo statico dei simboli per l'espansione a runtime |
| `repo_map.txt` | testo | Mappa del repository inserita nel prompt e passata al planner |
| `embedding_cache.sqlite` | SQLite | Vettori densi già calcolati, per hash di modello, lunghezza massima e testo del chunk |

### 12.3 Workspace (`esperto_workspace/`)

```
esperto_workspace/
├── current -> job-{uuid}/      # symlink atomico all'ultima estrazione attiva
└── job-{uuid}/
    └── …file estratti dallo ZIP…
```

Ogni upload produce una directory `job-{uuid}`; `_activate()` aggiorna il symlink `current` con `os.replace()` (operazione atomica su filesystem POSIX), solo dopo che il nuovo indice è stato attivato; le estrazioni precedenti vengono rimosse.

### 12.4 Ciclo di vita dei dati

1. **Ingestione**: ZIP → staging `job-{uuid}` → indicizzazione in una collection nuova → scambio dell'alias → symbol graph, mappa e stato persistiti → attivazione symlink.
2. **Query**: lettura da Qdrant, `symbol_graph.json` e `repo_map.txt`; nessuna scrittura salvo `last_prompt.json`.
3. **Re-upload identico**: fingerprint invariato → skip; l'indice attivo non viene toccato.
4. **Re-upload modificato**: fingerprint diverso → ricostruzione completa in una collection nuova; quella precedente resta servita fino allo scambio dell'alias ed è eliminata subito dopo.

---

## 13. Sicurezza

- **Validazione ZIP** ([zip_validation.py](esperto/src/ingestion/zip_validation.py)): magic bytes, integrità CRC, limite dimensione dell'archivio e del contenuto decompresso, rifiuto di path traversal e percorsi assoluti, whitelist di estensioni.
- **Grounding anti-allucinazione**: prompt con regole tassative (solo codice nel contesto, citazione obbligatoria delle fonti, divieto di assumere API inesistenti).
- **Validazione input a monte**: limiti di lunghezza sulle domande via Pydantic; sanitizzazione testo (`sanitize_text()`).
- **Isolamento dell'indicizzazione**: subprocess dedicato, staging separato, attivazione atomica.
- **Raggiungibilità ridotta**: la porta 5001 dell'indexer non è pubblicata sull'host.

Le limitazioni di sicurezza sono elencate in [§16](#16-limitazioni-note).

---

## 14. Testing

- **Unit test** (`esperto/tests/`, `benchmark/tests/`): `python -m pytest tests/ -v` dalla cartella del servizio (per `esperto`: `pip install -r requirements-dev.txt`). Nessuna dipendenza da Ollama o Qdrant: l'ambiente minimo è impostato nei rispettivi `conftest.py`.
- **Benchmark** (`benchmark/`, porta 5002, profilo compose `benchmark`): indicizza un repository a commit fissato, interroga `esperto` su un gold set (`datasets/<nome>/questions.yaml`) e produce in `benchmark_results/<run_id>/` metriche di retrieval, generazione e latenza con intervalli bootstrap. Endpoint, parametri e file prodotti sono in [§9.4](#94-benchmark--benchmark-5002).
  - **Ablation**: ogni configurazione disattiva un singolo stadio tramite `RetrievalOptions`.
  - **Campione stratificato**: 4 domande per categoria di default (`per_category`, `seed`), estratte in modo riproducibile.
  - **Piano condiviso**: nello stadio di retrieval il planner gira una volta per domanda e le configurazioni che lo condividono riusano le stesse sotto-domande (campo `subquestions` di `/retrieve`).
  - **Latenza**: i tempi (retrieval, primo token, totale) sono quelli delle richieste di generazione.
  - **Controlli**: il run fallisce subito se `esperto` non è pronto o se con `skip_index` l'indice attivo non è quello del dataset; il report mostra per ogni configurazione il numero di domande, le richieste fallite e la quota di risposte giudicate.
  - **Confronto tra run**: `GET /runs/{id}/compare/{altro}` calcola le differenze appaiate tra due run, per misurare l'effetto di una modifica al codice o all'indice.

---

## 15. Decisioni architetturali

| # | Decisione | Motivazione | Trade-off accettato |
|---|---|---|---|
| D1 | **Chunking AST-aware per Python** | Un chunk = un'unità semantica (funzione o classe; metodi solo per classi oltre 400 righe) con import di contesto: retrieval e citazioni più precise | Solo Python ha chunking strutturale; gli altri formati usano splitter testuale |
| D2 | **Ricerca densa più tabella dei simboli** | La ricerca densa trova per significato, anche tra lingue diverse (domande in italiano, codice in inglese); i nomi esatti li aggancia la tabella dei simboli | Nessuna ricerca lessicale: un termine raro che non sia il nome di un simbolo si trova solo se pesa nel significato del chunk |
| D3 | **Symbol graph statico da AST** | Espansione del contesto lungo relazioni reali del codice senza eseguirlo | Grafo limitato a relazioni staticamente rilevabili; utile solo se il dataset contiene Python |
| D4 | **Ingestion in subprocess con attivazione atomica** | Isolamento memoria/fallimenti; l'indice servito non è mai in stato intermedio (alias Qdrant) | Un job alla volta; latenza di avvio subprocess |
| D5 | **Fingerprint per idempotenza** | Re-upload dello stesso archivio è quasi gratuito; rebuild automatico al variare di modello o contenuti | Nessuna indicizzazione incrementale: ogni modifica ricostruisce tutto |
| D6 | **Streaming NDJSON con eventi tipizzati** | Fonti mostrate prima della generazione; UI progressiva; errori inline senza troncare | Il client deve implementare parsing line-delimited |
| D7 | **Config Pydantic centralizzata** | Errori di configurazione a boot-time, non a runtime | Ogni nuova opzione richiede modifica di `AppConfig` |
| D8 | **LLM locale via Ollama** | Privacy del codice (nulla esce dalla macchina), costo marginale nullo | Qualità dipendente dal modello locale; richiede GPU per buone prestazioni |
| D9 | **Frontend come proxy sottile** | Nessuna logica di business duplicata; un solo punto di verità (l'API esperto) | Il frontend eredita la latenza del backend |
| D10 | **Mappa del repository nel prompt e nel planner** | Visione d'insieme deterministica per le domande generali, dove la ricerca per somiglianza non basta | Occupa una quota fissa della finestra del modello a ogni richiesta |

---

## 16. Limitazioni note

- **Nessuna autenticazione/autorizzazione** su `/ask` e `/upload`: uso previsto solo in rete fidata.
- **Un solo job di indicizzazione alla volta** (mutex in-process): upload concorrenti ricevono `409`.
- **Nessuna indicizzazione incrementale**: ogni modifica ai sorgenti richiede un rebuild completo (mitigato dal fingerprint per i re-upload identici).
- **Chunking strutturale solo per Python**: altri linguaggi ricadono sullo splitter testuale.
- **Nessuna cache delle risposte**: ogni domanda riesegue retrieval e generazione completi.
- **Osservabilità minima**: logging applicativo e `last_prompt.json`, niente metriche/tracing strutturati.
- **Stato dei job in JSON**: nessuna persistenza storica dei job oltre l'ultimo.