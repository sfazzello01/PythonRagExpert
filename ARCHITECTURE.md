# ARCHITECTURE.md — Sistema "Esperto di Codebase"

> Documento di architettura del sistema RAG (Retrieval-Augmented Generation) per l'interrogazione di codebase Python.
> Ultimo aggiornamento: 2026-10-03.

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
16. [Limitazioni note e sviluppi futuri](#16-limitazioni-note-e-sviluppi-futuri)

---

## 1. Panoramica

**Esperto** è un assistente tecnico per codebase basato su **Retrieval-Augmented Generation**. L'utente carica l'archivio ZIP di un progetto software attraverso un'interfaccia web; il sistema lo indicizza in un database vettoriale e risponde a domande in linguaggio naturale con risposte generate da un LLM locale, **ancorate esclusivamente al codice indicizzato** e corredate dalle fonti (file, simbolo, righe).

### 1.1 Caratteristiche chiave

- **Retrieval ibrido a tre vie**: ricerca densa (BGE-M3, 1024 dimensioni), ricerca lessicale sparsa (BM25-like) e reranking token-level con vettori ColBERT, fusi con Reciprocal Rank Fusion (RRF).
- **Chunking semantico AST-aware**: i file Python sono suddivisi per funzione/classe tramite parsing AST, preservando gli import di contesto; fallback a chunking testuale per gli altri formati.
- **Symbol graph statico**: grafo di chiamate/ereditarietà/import costruito via AST (senza eseguire il codice) e usato a runtime per espandere il contesto con simboli correlati, tramite BFS bidirezionale e Personalized PageRank.
- **Planner LLM**: decomposizione opzionale della domanda in sotto-domande per interrogazioni multi-parte.
- **Ingestion asincrona**: upload ZIP validato lato sicurezza, estrazione in staging, indicizzazione in job di background isolato in subprocess, attivazione atomica via symlink.
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
| Embedding model | BGE-M3 (BAAI/bge-m3) via FlagEmbedding | 1.4.0 |
| Deep learning | PyTorch (CUDA 12.4) | 2.6.0+cu124 |
| Frontend | Jinja2 + HTML/JS (no framework) | 3.1.6 |
| Container | Docker + docker-compose | — |

---

## 2. Architettura di alto livello

Il sistema è composto da **cinque servizi** orchestrati con docker-compose. I tre servizi applicativi (`esperto`, `esperto_indexer`, `frontend_esperto`) condividono la stessa base di codice Python (i primi due) e comunicano via HTTP interno.

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
    IN -.->|"encode documenti (GPU/CPU)"| BGE

    BE --- ST[("esperto_state")]
    IN --- ST
    IN --- WS[("esperto_workspace")]
    BE --- WS
```

### 2.1 Ruoli e responsabilità

| Servizio | Immagine / Build | Porta | Ruolo |
|---|---|---|---|
| `frontend_esperto` | build `./frontend_esperto` | 8005 (esposta) | UI web: chat con streaming, upload ZIP, badge di stato sistema |
| `esperto` | build `./esperto` | 5000 (esposta) | API di interrogazione: `/ask` (streaming NDJSON) e `/health` |
| `esperto_indexer` | build `./esperto` (stessa immagine, comando diverso) | 5001 (solo interna) | Ingestion: `/upload`, `/status`, `/health`; esegue l'indicizzazione |
| `qdrant` | `qdrant/qdrant:v1.19.1` | 6333 | Vector database: collezione ibrida dense + sparse + ColBERT |
| `ollama` | `ollama/ollama:0.33.3` | 11434 | Runtime LLM locale per la generazione in streaming |

### 2.2 Separazione dei processi

`esperto` e `esperto_indexer` condividono codice e immagine Docker ma sono **processi separati per design**:

- L'**indexer** esegue il lavoro pesante e sincrono (parsing AST, embedding in batch, upsert) senza bloccare il servizio di query.
- Il **servizio di query** carica il modello di embedding solo per le query (lazy, su CPU di default) e mantiene latenze contenute.
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
| `ollama` | `./ollama_data:/root/.ollama` | `OLLAMA_CONTEXT_LENGTH=20000`, `OLLAMA_KV_CACHE_TYPE=q4_0`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1` | — | NVIDIA ×1 (opzionale) |
| `qdrant` | `./qdrant_data:/qdrant/storage` | — | TCP :6333, interval 10s, retries 12 | — |
| `esperto` | `./hf_cache:/root/.cache/huggingface`, `./esperto_state:/app/state`, `esperto_workspace:/app/workspace`, `./esperto_debug:/app/debug` | `.env` + `HF_HOME`, `PROMPT_DEBUG_FILE=/app/debug/last_prompt.json` | `curl -f http://127.0.0.1:5000/health`, start_period 30s, retries 20 | NVIDIA ×1 (opzionale) |
| `esperto_indexer` | `./hf_cache:/root/.cache/huggingface`, `./esperto_state:/app/state`, `esperto_workspace:/app/workspace` | `.env` + `HF_HOME`, `INGESTION_WORKSPACE_DIR=/app/workspace` | `curl -f http://127.0.0.1:5001/health`, start_period 15s | NVIDIA ×1 (opzionale) |
| `frontend_esperto` | — | `ESPERTO_URL=http://esperto:5000`, `ESPERTO_INDEXER_URL=http://esperto_indexer:5001`, `PYTHONPATH=/app` | — | — |

Volume condiviso dichiarato in compose: `esperto_workspace` (named volume, montato da entrambi i servizi `esperto*`).

### 3.2 Dipendenze di avvio

1. `ollama` e `qdrant` partono per primi (infrastruttura).
2. `esperto` attende `ollama` avviato e `qdrant` healthy; `esperto_indexer` idem.
3. `frontend_esperto` attende `esperto` avviato e `esperto_indexer` healthy.

### 3.3 Configurazione Ollama

Il container Ollama è configurato per un uso RAG intensivo:

- `OLLAMA_CONTEXT_LENGTH=20000`: finestra di contesto ampia, necessaria perché il prompt include fino a `RAG_CONTEXT_MAX_CHARS` (60000 caratteri) di contesto recuperato.
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
├── entrypoint.sh
├── pyproject.toml               # configurazione mypy
├── requirements.txt
└── src/
    ├── api.py                   # FastAPI :5000 — POST /ask (streaming), GET /health
    ├── rag.py                   # facade: stream_expert_response()
    ├── indicizzatore.py         # entry point indicizzazione: build_index()
    ├── core/
    │   ├── indexing/            # pipeline di indicizzazione
    │   │   ├── pipeline.py      #   orchestrazione: build_index(), _chunk_sources()
    │   │   ├── discovery.py     #   discovery file: iter_source_files(), index_targets(), service_for_path()
    │   │   ├── chunking.py      #   PythonASTContextualChunker + fallback_text_split()
    │   │   └── symbol_graph.py  #   costruzione grafo statico: build_symbol_graph(), save_symbol_graph()
    │   ├── rag/                 # pipeline di interrogazione
    │   │   ├── chain.py         #   stream_expert_response(): composizione retriever + LLM
    │   │   ├── retrieval.py     #   HybridRetriever: dense+sparse+ColBERT, enrichment, budget
    │   │   ├── planner.py       #   plan_subquestions(), parse_subquestions()
    │   │   ├── query.py         #   expand_queries()
    │   │   └── prompts.py       #   build_prompt(), INSTRUCTION (regole di grounding)
    │   ├── services/            # integrazione servizi esterni
    │   │   ├── qdrant.py        #   client sync/async, ensure_hybrid_collection(), hybrid_search(), upsert_encoded_documents()
    │   │   ├── embedding.py     #   ciclo di vita BGE-M3: load_indexing_model(), get_query_model(), encode_*()
    │   │   ├── index_state.py   #   fingerprint e stato: source_fingerprint(), is_current(), save()
    │   │   └── symbol_graph.py  #   SymbolGraph runtime: related_chunks() via BFS + PageRank
    │   └── shared/
    │       ├── settings.py      #   re-export config + costanti (DENSE_DIMENSION=1024, ...)
    │       ├── vectors.py       #   QueryVectors, lexical_weights_to_sparse_vector()
    │       └── text.py          #   sanitize_text()
    ├── ingestion/               # servizio di ingestion (:5001)
    │   ├── api.py               #   POST /upload, GET /status, GET /health
    │   ├── workspace.py         #   job in background, staging, attivazione atomica, subprocess
    │   └── zip_validation.py    #   validate_zip(): magic bytes, path traversal, integrità
    └── tests/                   # pytest: chunking, planner, symbol_graph, zip_validation
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

### 4.3 Moduli chiave e simboli principali

| Modulo | Simboli chiave | Responsabilità |
|---|---|---|
| [esperto/src/api.py](esperto/src/api.py) | `ChatRequest`, `ask_expert()`, `health_check()` | Contratto HTTP del servizio di query |
| [esperto/src/core/indexing/pipeline.py](esperto/src/core/indexing/pipeline.py) | `build_index()` | Orchestrazione end-to-end dell'indicizzazione |
| [esperto/src/core/indexing/chunking.py](esperto/src/core/indexing/chunking.py) | `PythonASTContextualChunker`, `fallback_text_split()` | Chunking semantico Python + fallback testuale |
| [esperto/src/core/indexing/discovery.py](esperto/src/core/indexing/discovery.py) | `iter_source_files()`, `index_targets()` | Scoperta dei file sorgente con filtri |
| [esperto/src/core/indexing/symbol_graph.py](esperto/src/core/indexing/symbol_graph.py) | `build_symbol_graph()`, `save_symbol_graph()` | Grafo statico di simboli/relazioni via AST |
| [esperto/src/core/rag/chain.py](esperto/src/core/rag/chain.py) | `stream_expert_response()` | Composizione retrieval → prompt → streaming LLM |
| [esperto/src/core/rag/retrieval.py](esperto/src/core/rag/retrieval.py) | `HybridRetriever`, `retrieve_many_with_sources()` | Retrieval ibrido, dedup, enrichment, budget contesto |
| [esperto/src/core/rag/planner.py](esperto/src/core/rag/planner.py) | `plan_subquestions()`, `parse_subquestions()` | Decomposizione LLM della domanda |
| [esperto/src/core/rag/prompts.py](esperto/src/core/rag/prompts.py) | `build_prompt()`, `INSTRUCTION` | Template di prompt con regole anti-allucinazione |
| [esperto/src/core/services/qdrant.py](esperto/src/core/services/qdrant.py) | `ensure_hybrid_collection()`, `hybrid_search()`, `upsert_encoded_documents()` | Integrazione Qdrant |
| [esperto/src/core/services/embedding.py](esperto/src/core/services/embedding.py) | `load_indexing_model()`, `get_query_model()` | Ciclo di vita BGE-M3 |
| [esperto/src/core/services/index_state.py](esperto/src/core/services/index_state.py) | `source_fingerprint()`, `is_current()` | Idempotenza dell'indice |
| [esperto/src/core/services/symbol_graph.py](esperto/src/core/services/symbol_graph.py) | `SymbolGraph.related_chunks()` | Espansione grafo a runtime (BFS + PageRank) |
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
    E --> F["_activate()<br/>symlink atomico current → staging"]
    F --> G["_run_indexer_subprocess()<br/>python -m src.indicizzatore<br/>env: BASE_DIR=staging, INDEX_TARGET_PATHS=."]

    subgraph IDX["Subprocess di indicizzazione (build_index)"]
        direction TB
        H["iter_source_files()<br/>discovery: SUPPORTED_EXTENSIONS, SKIP_DIRS"] --> I["source_fingerprint()<br/>SHA256 modello + contenuti"]
        I --> J{"is_current()?"}
        J -->|"sì: skip"| Z["touch index_complete<br/>job → ready"]
        J -->|"no"| K["ensure_hybrid_collection()<br/>dense + sparse + colbert"]
        K --> L["load_indexing_model()<br/>BGE-M3 (GPU se disponibile)"]
        L --> M["_chunk_sources()<br/>AST per .py, text splitter altrimenti"]
        M --> N["save_symbol_graph()<br/>nodi e archi via AST"]
        N --> O["batch: encode_documents()<br/>+ upsert_encoded_documents()"]
        O --> P["save()<br/>index_state.json + touch index_complete"]
        P --> Z
    end

    G --> H
    Z --> Q["index_jobs.json aggiornato<br/>status=ready, indexing_seconds"]
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

L'indicizzazione vera e propria non gira nel processo dell'ingestion service ma in un **subprocess** (`python -m src.indicizzatore`) con override delle variabili d'ambiente (`BASE_DIR` punta allo staging, `INDEX_TARGET_PATHS=.`). Questo garantisce:

- **Isolamento della memoria**: il modello BGE-M3 in modalità indexing (potenzialmente su GPU, FP16) viene caricato e rilasciato nel subprocess, senza occupare RAM/VRAM del servizio long-running.
- **Isolamento dei fallimenti**: un crash durante l'indicizzazione non abbatte l'ingestion service; lo stato del job riflette l'errore (`status=error` con coda dei log).
- **Configurazione puntuale**: l'indicizzazione avviene sempre contro lo staging appena attivato, indipendentemente dalla configurazione del servizio.

### 5.4 Discovery e fingerprint

`iter_source_files()` in [discovery.py](esperto/src/core/indexing/discovery.py) risolve i target (`INDEX_TARGET_PATHS`, relativi a `BASE_DIR`), applica il filtro sulle estensioni (`SUPPORTED_EXTENSIONS`) ed esclude le directory in `SKIP_DIRS` (`__pycache__`, `.git`, `node_modules`, …).

`source_fingerprint()` in [index_state.py](esperto/src/core/services/index_state.py) calcola un hash SHA256 che incorpora **sia la configurazione del modello** (nome, dimensione densa, lunghezza massima embedding) **sia il contenuto di ogni file**. Se il fingerprint coincide con quello persistito e la collection esiste già con il symbol graph, l'indicizzazione viene saltata (`is_current()` → `True`): caricare due volte lo stesso archivio è un'operazione a costo quasi nullo.

### 5.5 Chunking semantico

Definito in [chunking.py](esperto/src/core/indexing/chunking.py):

- **File Python** — `PythonASTContextualChunker`: parsing con `ast.parse()`, estrazione delle definizioni top-level (funzioni e classi) come chunk singoli, ciascuno arricchito con gli **import di contesto** del modulo e metadati ricchi (`filename`, `service`, `chunk_type`, `symbol_name`, `start_line`, `end_line`, `chunk_id`). Il corpo globale del modulo (codice fuori da definizioni) è preservato come chunk `global_body`. In caso di `SyntaxError` si ricade automaticamente sul fallback testuale.
- **Altri formati** — `fallback_text_split()`: `RecursiveCharacterTextSplitter` con `INDEX_CHUNK_SIZE=1200` caratteri e `INDEX_CHUNK_OVERLAP=150`, chunk marcati `text_block` con suffisso `part_N`.

Il `chunk_id` è deterministico e dipende dal tipo di chunk: per i simboli Python `MD5(filename::symbol_name::start_line-1::end_line)`, per i blocchi testuali `MD5(filename::part_N)`. Questo rende i punti Qdrant ricostruibili in modo stabile a ogni re-indicizzazione.

### 5.6 Costruzione del symbol graph

`build_symbol_graph()` in [indexing/symbol_graph.py](esperto/src/core/indexing/symbol_graph.py) esegue un'analisi **statica** dei file Python con due visitor AST:

- `_DefinitionVisitor`: raccoglie gli identificatori fully-qualified di moduli, classi, funzioni e metodi (nodi del grafo), ciascuno collegato al proprio `chunk_id`.
- `_GraphVisitor`: estrae gli archi tipizzati — `calls` (chiamate tra simboli), `inherits` (ereditarietà), `imports` — con la riga di origine.

Il grafo è salvato atomicamente in `esperto_state/symbol_graph.json` con un campo `version` (`SYMBOL_GRAPH_VERSION`) che invalida automaticamente grafi generati da versioni precedenti del formato.

### 5.7 Encoding e upsert

I documenti sono processati in batch (`BATCH_SIZE`, tipicamente 8): `encode_documents()` invoca BGE-M3 ottenendo per ogni chunk tre rappresentazioni (`dense_vecs`, `lexical_weights`, `colbert_vecs`), quindi `upsert_encoded_documents()` costruisce i `PointStruct` con ID deterministico `uuid5(NAMESPACE_URL, "esperto:{chunk_id}")` e li carica sulla collection. Gli upsert falliti per dimensione eccessiva ricadono su inserimenti individuali.

Al termine: `save()` scrive `index_state.json` (collection, fingerprint, numero file, versione grafo) e tocca il flag `index_complete`, che sblocca l'health check del servizio di query.

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
        BE->>PL: plan_subquestions(domanda) — temperature 0
        PL-->>BE: JSON {"subquestions": [...]} (2-4, fallback: domanda originale)
    end
    BE->>EM: encode_queries([domanda] + sotto-domande)
    EM-->>BE: QueryVectors {dense, sparse, colbert}
    loop ogni query di retrieval
        BE->>QD: hybrid_search: prefetch dense + sparse (RRF) → rerank ColBERT
        QD-->>BE: candidati con _colbert_score
    end
    BE->>BE: dedup per chunk_id, enrichment same-file
    BE->>SG: related_chunks(chunk_ids) — BFS + Personalized PageRank
    SG-->>BE: chunk correlati + relazioni
    BE->>QD: get_documents_by_chunk_ids(correlati)
    QD-->>BE: documenti arricchiti (_graph_relation)
    BE->>BE: _apply_context_budget (≤35 fonti, ≤60k caratteri)
    BE-->>FE: evento NDJSON "sources" (fonti + sotto-domande)
    FE-->>U: rendering elenco fonti
    BE->>BE: build_prompt(contesto, domanda) + salvataggio debug
    BE->>LL: astream(prompt) — ChatOllama
    loop streaming
        LL-->>BE: chunk (answer / reasoning_content)
        BE-->>FE: eventi NDJSON "answer" / "reasoning"
        FE-->>U: rendering incrementale (markdown)
    end
```

### 6.2 Planner delle sotto-domande

Con `use_planner=true` (default nel frontend), `plan_subquestions()` in [planner.py](esperto/src/core/rag/planner.py) interroga l'LLM con temperatura 0 chiedendo la decomposizione in 2-4 sotto-domande autonome. Il prompt del planner vieta esplicitamente di inventare nomi di file o tecnologie. `parse_subquestions()` valida il JSON prodotto, deduplica case-insensitive, filtra frammenti troppo corti e, in caso di output non valido, **ricade sulla domanda originale**: il planner non può mai peggiorare il comportamento di base.

Al retrieval partecipano la domanda originale più al massimo 4 sotto-domande uniche, processate in parallelo.

### 6.3 Retrieval ibrido a tre vie

Implementato da `HybridRetriever` in [retrieval.py](esperto/src/core/rag/retrieval.py) e `hybrid_search()` in [services/qdrant.py](esperto/src/core/services/qdrant.py):

1. **Prefetch parallelo**: ricerca densa (HNSW su vettore `dense`) e ricerca sparsa (vettore `sparse`, pesi lessicali BGE-M3), ciascuna con `PREFETCH_LIMIT` candidati (valore tipico 20).
2. **Fusione RRF**: Reciprocal Rank Fusion combina i ranking denso e sparso senza richiedere normalizzazione degli score.
3. **Reranking ColBERT**: i primi `RERANK_CANDIDATE_LIMIT=50` candidati fusi sono ri-ordinati con similarità MaxSim token-level (multi-vettore `colbert`, residente su disco e calcolato a query-time). Lo score finale è esposto come `_colbert_score`.

### 6.4 Post-processing del contesto

Sui risultati aggregati di tutte le query si applicano in sequenza:

1. **Deduplica** per `chunk_id` mantenendo lo score massimo (`_keep_highest_scoring()`).
2. **Enrichment same-file**: per ogni file che compare per la prima volta nei top risultati, fino a 4 chunk aggiuntivi dallo stesso file per fornire contesto locale.
3. **Espansione via symbol graph**: i `chunk_id` recuperati fungono da semi per `SymbolGraph.related_chunks()`, che restituisce fino a `SYMBOL_GRAPH_MAX_NEIGHBORS=6` chunk correlati (funzioni chiamanti/chiamate, superclassi/sottoclassi) con la relazione esposta in `_graph_relation`. Vedi [§8](#8-symbol-graph).
4. **Budget di contesto** (`_apply_context_budget()`): ordinamento per score decrescente e selezione entro i limiti `RAG_CONTEXT_MAX_SOURCES=35` fonti e `RAG_CONTEXT_MAX_CHARS=60000` caratteri complessivi.

### 6.5 Prompt e grounding

`build_prompt()` in [prompts.py](esperto/src/core/rag/prompts.py) compone un `SystemMessage` con le regole e un `HumanMessage` con contesto + domanda. Le regole tassative includono:

- usare **esclusivamente** il codice nel contesto fornito; se l'informazione non è visibile, dichiararlo esplicitamente;
- non assumere l'esistenza di librerie o funzioni non presenti;
- citare sempre file e funzione/linea di riferimento;
- per le modifiche proposte, mostrare "Prima" e "Dopo";
- concisione, niente parafrasi ripetitive né chiusure generiche.

Il prompt renderizzato di ogni richiesta è salvato in `esperto_debug/last_prompt.json` (`PROMPT_DEBUG_FILE`) per ispezione.

### 6.6 Streaming e gestione errori

`stream_expert_response()` in [chain.py](esperto/src/core/rag/chain.py) è un async generator che emette nell'ordine:

| `type` | `data` | Quando |
|---|---|---|
| `sources` | `{sources: [...], subquestions: [...]}` | Una volta, prima della generazione |
| `reasoning` | frammento di reasoning | Solo se `use_reasoning=true` e il modello lo produce |
| `answer` | frammento di risposta | Streaming incrementale del completamento |

Le fonti includono `filename`, `service`, `chunk_type`, `symbol_name`, `start_line`, `end_line`, `score` ed eventualmente `graph_relation`. Gli errori a metà stream (timeout LLM oltre `LLM_INFERENCE_TIMEOUT`, eccezioni generiche) sono convertiti in eventi `[ERROR] ...` invece di troncare silenziosamente la risposta.

---

## 7. Modello dati Qdrant

### 7.1 Spazi vettoriali della collection

La collection (nome da `QDRANT_COLLECTION`, default d'uso `codebase_expert_bgem3_v1`) è **ibrida a tre spazi**, creata da `ensure_hybrid_collection()` in [services/qdrant.py](esperto/src/core/services/qdrant.py):

| Vettore | Dimensione | Distanza | Indicizzazione | Storage | Quantizzazione | Ruolo |
|---|---|---|---|---|---|---|
| `dense` | 1024 | COSINE | HNSW | RAM | INT8 scalare (quantile 0.99, sempre in RAM) | Similarità semantica a livello di chunk |
| `sparse` | variabile | — | indice sparse dedicato | RAM | — | Matching lessicale BM25-like (da `lexical_weights` BGE-M3) |
| `colbert` | 1024 × N token | COSINE (MaxSim multi-vettore) | nessun HNSW | disco | INT8 (quantile 0.99, non residente in RAM) | Reranking fine token-level a query-time |

La strategia complessiva è: **recall** da dense+sparse (con fusione RRF), **precisione** dal reranking ColBERT. La scelta di tenere il multi-vettore ColBERT su disco senza indice HNSW è deliberata: viene interrogato solo su ~50 candidati già filtrati, quindi il costo del confronto esaustivo è contenuto e si risparmia la RAM che un indice su multi-vettori richiederebbe.

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

Valori di `chunk_type`: `function`, `class`, `global_body`, `text_block`. Il campo `service` è derivato dal percorso (`service_for_path()`) e usato nei metadati esposti al client.

### 7.3 Identificatori deterministici

- **`chunk_id`**: deterministico — `MD5(filename::symbol_name::start_line-1::end_line)` per simboli Python, `MD5(filename::part_N)` per blocchi testuali.
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

`related_chunks(chunk_ids, limit)` esegue:

1. **BFS bidirezionale** dagli archi `calls`/`inherits` a partire dai simboli seed, con profondità massima `SYMBOL_GRAPH_MAX_DEPTH=3`, escludendo i seed stessi e i chunk già presenti nel contesto.
2. **Personalized PageRank** sui candidati raggiunti (damping `0.4`, 20 iterazioni, tolleranza `1e-6`), personalizzato sui seed: misura la "vicinanza strutturale" nel grafo, non la semplice distanza in archi.
3. **Selezione** dei top `SYMBOL_GRAPH_MAX_NEIGHBORS=6` chunk correlati, ciascuno annotato con il percorso BFS più breve come stringa di relazione (esposta al client come `graph_relation`).

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

**Response**: `StreamingResponse`, `text/plain; charset=utf-8`, una riga JSON per evento:

```json
{"type": "sources", "data": {"sources": [{"filename": "…", "symbol_name": "…", "chunk_type": "…", "score": 0.83}], "subquestions": ["…"]}}
{"type": "reasoning", "data": "…frammento di ragionamento…"}
{"type": "answer", "data": "…frammento di risposta…"}
```

**Errori**: `400` messaggio non valido; `500` errore interno; eventi `[ERROR]` inline per timeout/errori a stream avviato.

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
| `MODEL_LOAD_TIMEOUT` | obbligatoria | ≥ 10s | Timeout caricamento modelli |

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
| `K` | obbligatoria | 1–50 | Top-K risultati finali |
| `PREFETCH_LIMIT` | obbligatoria | 1–100 | Candidati prefetch per dense e sparse |
| `SEMANTIC_RELEVANCE_RATIO` | obbligatoria | 0.0–1.0 | Soglia minima di rilevanza relativa |
| `RAG_CONTEXT_MAX_CHARS` | 60000 | ≥ 1 | Budget caratteri del contesto |
| `RAG_CONTEXT_MAX_SOURCES` | 35 | ≥ 1 | Numero massimo di fonti nel contesto |
| `MIN_MESSAGE_LENGTH` / `MAX_MESSAGE_LENGTH` | obbligatorie | 1–100 / 100–50000 | Limiti lunghezza domanda (valori tipici 3 / 5000) |

### 11.4 Embedding (BGE-M3)

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `DENSE_MODEL_NAME` | `BAAI/bge-m3` | — | Identificativo HuggingFace del modello |
| `BGE_M3_DEVICE` | obbligatoria | `cuda` / `cpu` | Device per l'indicizzazione (le query usano CPU) |
| `BGE_M3_USE_FP16` | obbligatoria | bool | Half precision su GPU |
| `EMBEDDING_MAX_LENGTH` | obbligatoria | 128–8192 | Lunghezza massima in token per encoding (tipicamente 1024) |
| `BATCH_SIZE` | obbligatoria | 1–128 | Batch size dell'encoding in indicizzazione (tipicamente 8) |

### 11.5 Indicizzazione e discovery

| Variabile | Default | Descrizione |
|---|---|---|
| `BASE_DIR` | obbligatoria | Radice dei sorgenti da indicizzare (nel compose: `/app/workspace/current`) |
| `INDEX_TARGET_PATHS` | obbligatoria | Path relativi a `BASE_DIR` (CSV) inclusi nella discovery |
| `SUPPORTED_EXTENSIONS` | obbligatoria | Estensioni indicizzabili (tipiche: `.py,.yaml,.yml,.md,.txt,.html,.json,.js`) |
| `SKIP_DIRS` | obbligatoria | Directory escluse (tipiche: `__pycache__,.git,node_modules,…`) |
| `INDEX_CHUNK_SIZE` / `INDEX_CHUNK_OVERLAP` | 1200 / 150 | Dimensione e overlap del fallback testuale |
| `INDEX_STATE_DIR` | obbligatoria | Directory dei file di stato (nel compose: `/app/state`) |

### 11.6 Symbol graph

| Variabile | Default | Vincoli | Descrizione |
|---|---|---|---|
| `SYMBOL_GRAPH_MAX_NEIGHBORS` | 6 | ≥ 1 | Chunk correlati massimi per espansione |
| `SYMBOL_GRAPH_MAX_DEPTH` | 3 | ≥ 0 | Profondità BFS |
| `SYMBOL_GRAPH_PAGERANK_DAMPING` | 0.4 | (0.0, 1.0) | Damping factor PageRank |
| `SYMBOL_GRAPH_PAGERANK_ITERATIONS` | 20 | ≥ 1 | Iterazioni PageRank |
| `SYMBOL_GRAPH_PAGERANK_TOLERANCE` | 1e-6 | > 0 | Tolleranza di convergenza |

### 11.7 Ingestion e debug

| Variabile | Default | Descrizione |
|---|---|---|
| `MAX_UPLOAD_MB` | 200 | Dimensione massima dello ZIP |
| `INGESTION_WORKSPACE_DIR` | `/app/workspace` | Directory di staging degli upload |
| `PROMPT_DEBUG_FILE` | `/app/debug/last_prompt.json` | Salvataggio dell'ultimo prompt renderizzato |

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

### 12.2 File di stato (`esperto_state/`)

| File | Contenuto | Scopo |
|---|---|---|
| `index_state.json` | `{collection, fingerprint, source_file_count, symbol_graph_version}` | Idempotenza: confronto fingerprint per skip del rebuild |
| `index_complete` | file flag vuoto | Segnala indice utilizzabile; letto da `/health` |
| `index_jobs.json` | `{job_id, filename, status, files, started_at, finished_at, error, indexing_seconds}` | Stato dell'ultimo job di upload (`queued`/`indexing`/`ready`/`error`) |
| `symbol_graph.json` | `{version, nodes[], edges[]}` | Grafo statico dei simboli per l'espansione a runtime |

### 12.3 Workspace (`esperto_workspace/`)

```
esperto_workspace/
├── current -> job-{uuid}/      # symlink atomico all'ultima estrazione attiva
└── job-{uuid}/
    └── …file estratti dallo ZIP…
```

Ogni upload produce una directory `job-{uuid}`; `_activate()` aggiorna il symlink `current` con `os.replace()` (operazione atomica su filesystem POSIX), così il servizio di query vede sempre una workspace consistente anche durante una re-indicizzazione.

### 12.4 Ciclo di vita dei dati

1. **Ingestione**: ZIP → staging `job-{uuid}` → attivazione symlink → indicizzazione → stato persistito.
2. **Query**: lettura da Qdrant e `symbol_graph.json`; nessuna scrittura salvo `last_prompt.json`.
3. **Re-upload identico**: fingerprint invariato → skip; la collection non viene toccata.
4. **Re-upload modificato**: fingerprint diverso → ricostruzione completa con ID punto deterministici (upsert idempotente sulle stesse chiavi).

---

## 13. Sicurezza

### 13.1 Controlli implementati

- **Validazione ZIP** ([zip_validation.py](esperto/src/ingestion/zip_validation.py)): magic bytes, integrità CRC, limite dimensione, rifiuto di path traversal e percorsi assoluti, whitelist di estensioni.
- **Grounding anti-allucinazione**: prompt con regole tassative (solo codice nel contesto, citazione obbligatoria delle fonti, divieto di assumere API inesistenti).
- **Validazione input a monte**: limiti di lunghezza sulle domande via Pydantic; sanitizzazione testo (`sanitize_text()`).
- **Isolamento dell'indicizzazione**: subprocess dedicato, staging separato, attivazione atomica.
- **Raggiungibilità ridotta**: la porta 5001 dell'indexer non è pubblicata sull'host.

### 13.2 Limitazioni note

- **Nessuna autenticazione** su `/ask` e `/upload`: il sistema è pensato per uso locale/interno, non per esposizione pubblica.
- **Un solo job di indicizzazione alla volta** (mutex in-process): upload concorrenti ricevono `409`.

---

## 15. Decisioni architetturali

| # | Decisione | Motivazione | Trade-off accettato |
|---|---|---|---|
| D1 | **Chunking AST-aware per Python** | Un chunk = un'unità semantica (funzione/classe) con import di contesto: retrieval e citazioni più precise | Solo Python ha chunking strutturale; gli altri formati usano splitter testuale |
| D2 | **Retrieval ibrido a tre vie (dense + sparse + ColBERT)** | La densa cattura semantica, la sparsa i termini esatti (nomi di simboli!), ColBERT rifinisce token-level | Costo computazionale a query-time per il reranking; multi-vettore su disco per contenerlo |
| D3 | **Symbol graph statico da AST** | Espansione del contesto lungo relazioni reali del codice senza eseguirlo | Grafo limitato a relazioni staticamente rilevabili; utile solo se il dataset contiene Python |
| D4 | **Ingestion in subprocess con attivazione atomica** | Isolamento memoria/fallimenti; la workspace non è mai in stato intermedio | Un job alla volta; latenza di avvio subprocess |
| D5 | **Fingerprint per idempotenza** | Re-upload dello stesso archivio è quasi gratuito; rebuild automatico al variare di modello o contenuti | Nessuna indicizzazione incrementale: ogni modifica ricostruisce tutto |
| D6 | **Streaming NDJSON con eventi tipizzati** | Fonti mostrate prima della generazione; UI progressiva; errori inline senza troncare | Il client deve implementare parsing line-delimited |
| D7 | **Config Pydantic centralizzata** | Errori di configurazione a boot-time, non a runtime | Ogni nuova opzione richiede modifica di `AppConfig` |
| D8 | **LLM locale via Ollama** | Privacy del codice (nulla esce dalla macchina), costo marginale nullo | Qualità dipendente dal modello locale; richiede GPU per buone prestazioni |
| D9 | **Frontend come proxy sottile** | Nessuna logica di business duplicata; un solo punto di verità (l'API esperto) | Il frontend eredita la latenza del backend |

---

## 16. Limitazioni note

### 16.1 Limitazioni attuali

- **Nessuna autenticazione/autorizzazione**: uso previsto solo in rete fidata.
- **Nessuna indicizzazione incrementale**: ogni modifica ai sorgenti richiede un rebuild completo (mitigato dal fingerprint per i re-upload identici).
- **Query expansion placeholder**: `expand_queries()` è attualmente un passthrough; la diversificazione delle query è affidata al solo planner.
- **Chunking strutturale solo per Python**: altri linguaggi ricadono sullo splitter testuale.
- **Nessuna cache delle risposte**: ogni domanda riesegue retrieval e generazione completi.
- **Osservabilità minima**: logging applicativo e `last_prompt.json`, niente metriche/tracing strutturati.
- **Stato dei job in JSON**: nessuna persistenza storica dei job oltre l'ultimo.