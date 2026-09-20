# Esperto RAG: flussi completi del sistema

Questo documento descrive il funzionamento dell'Esperto seguendo il codice attuale.
Il sistema ha due percorsi principali:

1. **Indicizzazione**: riceve uno ZIP, prepara una workspace, crea chunk, embedding e symbol graph, poi salva tutto in Qdrant.
2. **Domanda/RAG**: riceve una domanda, eventualmente genera sotto-domande, recupera il contesto, costruisce il prompt e trasmette la risposta LLM in streaming.

La directory `esperto/src/core` contiene la logica principale. Il file `backend/src/crawler/backend.py` non partecipa a questi flussi: appartiene al crawler del backend applicativo.

---

## 1. Architettura generale

```text
CLIENT
  |
  +--> POST /upload :5001
  |       |
  |       +--> valida ZIP
  |       +--> crea job
  |       +--> workspace staging
  |       +--> indicizzatore subprocess
  |       +--> Qdrant + symbol_graph.json
  |
  +--> GET /status :5001
  |
  +--> POST /ask :5000
          |
          +--> valida domanda
          +--> planner LLM opzionale
          +--> retrieval ibrido
          |      +--> dense prefetch
          |      +--> sparse prefetch
          |      +--> ColBERT reranking
          |      +--> deduplica
          |      +--> stesso-file expansion
          |      +--> symbol graph expansion (BFS 3 hop + PPR diretto)
          |      +--> reranking globale ColBERT + filtro soglia relativa
          |      +--> context budget
          +--> invia sources
          +--> costruisce prompt grounding
          +--> genera risposta con Ollama
          +--> stream di reasoning/answer
```

Servizi Docker rilevanti:

- `esperto`: API RAG su porta `5000`.
- `esperto_indexer`: API upload/status su porta `5001`.
- `qdrant`: database vettoriale su porta `6333`.
- `ollama`: modello LLM usato dal planner e dalla risposta finale.

Configurazione dei container e dei volumi: [docker-compose.yaml](docker-compose.yaml).

---

## 2. Flusso di upload e indicizzazione

### 2.1 Ricezione dello ZIP

L'ingresso è:

```text
POST /upload :5001
```

Implementazione: [src/ingestion/api.py](esperto/src/ingestion/api.py), funzione `upload_zip()`.

Il servizio:

1. legge il file tramite `UploadFile`;
2. legge tutto il contenuto in memoria;
3. chiama `validate_zip()`;
4. avvia un job asincrono in background con `start_index_job()`;
5. restituisce subito HTTP `202`.

Risposta concettuale:

```json
{
  "status": "queued",
  "job": {
    "job_id": "...",
    "filename": "codebase.zip",
    "status": "queued"
  }
}
```

Il caricamento non esegue l'indicizzazione dentro la richiesta HTTP: il lavoro continua in un thread daemon.

### 2.2 Validazione ZIP

Implementazione: [src/ingestion/zip_validation.py](esperto/src/ingestion/zip_validation.py), funzione `validate_zip()`.

Controlli eseguiti:

- il nome termina con `.zip`;
- il file non è vuoto;
- la dimensione non supera `MAX_UPLOAD_MB`;
- la firma binaria è compatibile con ZIP;
- l'archivio è leggibile;
- il checksum delle entry è valido;
- non sono ammessi path assoluti;
- non sono ammessi path traversal come `../`;
- sono considerate solo le estensioni configurate in `SUPPORTED_EXTENSIONS`.

I file con estensione non supportata vengono ignorati. L'archivio viene rifiutato se non rimane almeno un file indicizzabile.

Il risultato della validazione è la lista ordinata dei nomi ammessi, non l'estrazione dei file.

### 2.3 Creazione e stato del job

Implementazione: [src/ingestion/workspace.py](esperto/src/ingestion/workspace.py), funzione `start_index_job()`.

Gli stati possibili sono:

```text
not_indexed -> queued -> indexing -> ready
                              |
                              +--> error
```

Il job viene salvato in `INDEX_STATE_DIR/index_jobs.json`.

Se esiste già un job in stato `queued` o `indexing`, un nuovo upload viene rifiutato con errore HTTP `409`.

L'endpoint:

```text
GET /status :5001
```

chiama `current_job_status()` e restituisce lo stato persistito.

### 2.4 Estrazione nella staging directory

Implementazione: [src/ingestion/workspace.py](esperto/src/ingestion/workspace.py), funzione `extract_to_workspace()`.

Per ogni upload viene creata una directory:

```text
WORKSPACE_ROOT/job-<id>
```

I file vengono estratti solo se compaiono nella lista già validata. Le directory parent vengono create prima della copia.

Successivamente `_activate()` aggiorna atomicamente il link:

```text
WORKSPACE_ROOT/current -> nuova staging directory
```

Il link viene sostituito tramite un link temporaneo e `os.replace()`, così il prossimo processo vede una workspace coerente.

### 2.5 Avvio dell'indicizzatore

Implementazione: [src/ingestion/workspace.py](esperto/src/ingestion/workspace.py), funzione `_run_indexer_subprocess()`.

L'indicizzatore viene lanciato come processo separato:

```text
python -m src.indicizzatore
```

Vengono sovrascritte le variabili:

```text
BASE_DIR=<staging directory>
INDEX_TARGET_PATHS=.
```

Il nuovo processo è necessario perché la configurazione viene letta a import-time da `config.py`.

Il comando invoca [src/indicizzatore.py](esperto/src/indicizzatore.py), che è una facciata verso:

```python
from .core.indexing.pipeline import build_index
```

A processo terminato:

- codice di ritorno `0` -> job `ready`;
- altro codice di ritorno -> job `error` con coda dei log;
- eccezione -> job `error` con messaggio dell'eccezione.

---

## 3. Pipeline di indicizzazione

Implementazione principale: [src/core/indexing/pipeline.py](esperto/src/core/indexing/pipeline.py), funzione `build_index()`.

Il flusso è:

```text
wait_for_qdrant()
  -> iter_source_files()
  -> source_fingerprint()
  -> controllo indice già aggiornato
  -> clear()
  -> ensure_hybrid_collection()
  -> load_indexing_model()
  -> _chunk_sources()
  -> save_symbol_graph()
  -> encode_documents()
  -> upsert_encoded_documents()
  -> save()
```

### 3.1 Attesa di Qdrant

`wait_for_qdrant()` in [src/core/services/qdrant.py](esperto/src/core/services/qdrant.py) prova a contattare Qdrant prima di iniziare.

Se Qdrant non risponde entro i tentativi previsti, l'indicizzazione fallisce.

### 3.2 Discovery dei file

Implementazione: [src/core/indexing/discovery.py](esperto/src/core/indexing/discovery.py).

`index_targets()` risolve `INDEX_TARGET_PATHS` rispetto a `BASE_DIR`.

`iter_source_files()`:

1. accetta file singoli o directory;
2. attraversa ricorsivamente le directory;
3. filtra per `SUPPORTED_EXTENSIONS`;
4. esclude le directory presenti in `SKIP_DIRS`;
5. restituisce i file ordinati deterministicamente.

`service_for_path()` assegna come servizio il primo segmento del path relativo. Per esempio:

```text
backend/src/api.py -> service = backend
```

### 3.3 Fingerprint e riuso dell'indice

La pipeline calcola un fingerprint dei file sorgente tramite `source_fingerprint()` in [src/core/services/index_state.py](esperto/src/core/services/index_state.py).

Se tutte le condizioni sono vere:

```text
collection Qdrant esistente
+ fingerprint invariato
+ versione symbol graph compatibile
+ stato persistente presente
```

l'indicizzazione viene saltata e viene solo aggiornato/toccato il flag `index_complete`.

Se l'indice non è aggiornato, `clear()` elimina lo stato persistente precedente e il contenuto viene ricostruito.

### 3.4 Creazione della collection Qdrant

Implementazione: [src/core/services/qdrant.py](esperto/src/core/services/qdrant.py), funzione `ensure_hybrid_collection()`.

La collection contiene tre spazi vettoriali:

```python
vectors_config={
    "dense": VectorParams(..., distance=Distance.COSINE),
    "colbert": VectorParams(
        ...,
        multivector_config=MultiVectorConfig(
            comparator=MultiVectorComparator.MAX_SIM
        ),
    ),
}
sparse_vectors_config={"sparse": SparseVectorParams()}
```

Le dimensioni e i parametri dipendono da BGE-M3 e dalla configurazione.

### 3.5 Chunking dei file

Implementazione: [src/core/indexing/pipeline.py](esperto/src/core/indexing/pipeline.py), `_chunk_sources()`.

Il comportamento dipende dall'estensione:

```text
.py       -> PythonASTContextualChunker
altri     -> fallback_text_split
```

#### File Python

Implementazione: [src/core/indexing/chunking.py](esperto/src/core/indexing/chunking.py), classe `PythonASTContextualChunker`.

Il codice viene analizzato con `ast.parse()`.

Per ogni funzione, funzione asincrona o classe top-level viene creato un `Document` con:

- filename relativo;
- servizio;
- tipo chunk (`function`, `class`);
- nome simbolo;
- linea iniziale e finale;
- `chunk_id` deterministico;
- import del modulo;
- codice sorgente del simbolo;
- eventuale decoratore API.

Il contenuto contiene sezioni esplicite:

```text
# FILE: ...
# TIPO: FUNCTION/CLASS
--- IMPORTS DI CONTESTO ---
--- CODICE SORGENTE ---
```

Se esiste codice globale non già coperto da funzioni/classi e supera la soglia minima, viene creato anche un chunk `global_body`.

Se il parsing AST fallisce per un `SyntaxError`, il file passa al fallback testuale.

#### File non Python

`fallback_text_split()` usa `RecursiveCharacterTextSplitter` con:

```text
INDEX_CHUNK_SIZE = 1200
INDEX_CHUNK_OVERLAP = 150
```

Ogni chunk viene marcato come `text_block` e riceve un `chunk_id` deterministico basato sul file e sull'indice della parte.

### 3.6 Creazione del symbol graph

Implementazione: [src/core/indexing/symbol_graph.py](esperto/src/core/indexing/symbol_graph.py).

`save_symbol_graph(source_files)` analizza solo i file `.py`, senza importarli né eseguirli.

Il processo è:

```text
collect definition IDs
  -> visita AST dei file Python
  -> crea nodi module/class/function/method
  -> risolve riferimenti locali e importati
  -> crea edge calls/inherits/imports/defines
  -> deduplica gli edge
  -> scrive symbol_graph.json atomically
```

Ogni nodo contiene:

```text
id, kind, name, filename, start_line, end_line, chunk_id
```

Gli edge contengono:

```text
source, target, kind, line
```

I chunk sono associati soprattutto ai simboli top-level. I metodi annidati possono esistere come nodi ma non avere un chunk indipendente.

Il file viene scritto in:

```text
INDEX_STATE_DIR/symbol_graph.json
```

### 3.7 Embedding dei chunk

Implementazione: [src/core/services/embedding.py](esperto/src/core/services/embedding.py).

`load_indexing_model()` carica BGE-M3:

```text
BAAI/bge-m3
```

`encode_documents()` chiede al modello tutte le rappresentazioni:

```python
return_dense=True
return_sparse=True
return_colbert_vecs=True
```

Per ogni chunk vengono prodotti:

```text
dense   -> un vettore semantico
sparse  -> lexical weights convertiti in SparseVector
colbert -> lista di vettori token-level
```

### 3.8 Upsert in Qdrant

Implementazione: [src/core/services/qdrant.py](esperto/src/core/services/qdrant.py), `upsert_encoded_documents()`.

Ogni `Document` viene salvato come punto Qdrant:

```python
vector={
    "dense": dense_vector,
    "sparse": sparse_vector,
    "colbert": colbert_vectors,
}
payload={
    "page_content": document.page_content,
    "metadata": document.metadata,
}
```

L'ID Qdrant è UUID deterministico derivato dal `chunk_id` applicativo.

L'upsert avviene a batch. Se il payload complessivo è troppo grande, il codice riprova con upsert individuali.

### 3.9 Chiusura dell'indicizzazione

Dopo tutti i batch, `save()` scrive lo stato in [src/core/services/index_state.py](esperto/src/core/services/index_state.py), includendo fingerprint, numero file e versione del symbol graph.

Il flag `index_complete` consente all'API RAG di distinguere un indice pronto da uno non ancora creato.

---

## 4. Flusso della domanda `/ask`

Ingresso: [src/api.py](esperto/src/api.py), funzione `ask_expert()`.

Endpoint:

```text
POST /ask :5000
Content-Type: application/json
```

Payload concettuale:

```json
{
  "message": "Come viene creato l'indice?",
  "use_planner": true,
  "use_reasoning": false
}
```

`ChatRequest` applica i limiti min/max configurati. Dopo la validazione Pydantic, il codice esegue `strip()` e rifiuta una domanda composta solo da spazi.

La risposta è una `StreamingResponse` con media type:

```text
text/plain; charset=utf-8
```

Ogni riga è un JSON indipendente.

### 4.1 Wrapper di timeout e errori

`ask_expert()` crea `timeout_wrapper()` intorno a `stream_expert_response()`.

Se si verifica un timeout LLM, viene trasmesso:

```text
[ERROR] La risposta ha superato il timeout massimo.
```

Per altre eccezioni viene trasmesso il tipo dell'errore.

### 4.2 Entry point RAG

Implementazione: [src/core/rag/chain.py](esperto/src/core/rag/chain.py), funzione `stream_expert_response()`.

Il flusso è:

```text
stream_expert_response()
  -> plan_subquestions() oppure [question]
  -> HybridRetriever.retrieve_many_with_sources()
  -> yield sources
  -> build_prompt().invoke(...)
  -> salva prompt di debug opzionale
  -> ChatOllama.astream(...)
  -> yield reasoning e answer
```

Il retriever viene creato lazy come singleton tramite `_get_retriever()`, evitando di aprire il client Qdrant al momento dell'import del modulo.

---

## 5. Planner delle sotto-domande

Implementazione: [src/core/rag/planner.py](esperto/src/core/rag/planner.py).

Quando `use_planner=true`, il planner usa un'istanza `ChatOllama` separata con:

```text
temperature = 0
reasoning = false
```

Il prompt chiede da due a quattro sotto-domande solo quando aiutano il retrieval. Una domanda già specifica può rimanere singola.

Il risultato atteso è esclusivamente:

```json
{"subquestions": ["...", "..."]}
```

`parse_subquestions()`:

1. cerca il primo oggetto JSON nella risposta;
2. legge `subquestions`;
3. accetta al massimo quattro elementi;
4. elimina elementi non stringa;
5. normalizza il testo;
6. elimina duplicati case-insensitive;
7. usa la domanda originale come fallback se il JSON è invalido o vuoto.

Se la chiamata al planner fallisce, `plan_subquestions()` restituisce direttamente `[question]`.

Nel caso `use_planner=false`, il planner viene saltato e il retrieval riceve una sola domanda.

---

## 6. Retrieval ibrido: dense, sparse e ColBERT

Il percorso principale è nella classe `HybridRetriever` in [src/core/rag/retrieval.py](esperto/src/core/rag/retrieval.py).

### 6.1 Espansione iniziale della query

`retrieve_with_sources()` chiama `expand_queries()` in [src/core/rag/query.py](esperto/src/core/rag/query.py).

Nel codice attuale la funzione restituisce la domanda sanitizzata. La seconda variante è commentata, quindi di fatto non è attiva una vera query expansion automatica.

### 6.2 Encoding della query

`encode_queries()` in [src/core/services/embedding.py](esperto/src/core/services/embedding.py) genera per ogni query:

```text
dense query vector
sparse query vector
ColBERT query multivector
```

Il risultato è rappresentato da `QueryVectors` in [src/core/shared/vectors.py](esperto/src/core/shared/vectors.py).

### 6.3 Prefetch dense e sparse

`retrieve_with_sources()` esegue `hybrid_search()` per ogni query.

Implementazione: [src/core/services/qdrant.py](esperto/src/core/services/qdrant.py), funzione `hybrid_search()`.

La richiesta a Qdrant contiene due prefetch distinti:

```python
prefetch=[
    Prefetch(query=query_vectors.dense, using="dense", limit=PREFETCH_LIMIT),
    Prefetch(query=query_vectors.sparse, using="sparse", limit=PREFETCH_LIMIT),
]
```

Dense e sparse producono candidati complementari:

- dense cerca similarità semantica;
- sparse privilegia termini e segnali lessicali;
- un chunk trovato da entrambi può comparire nello stesso pool di candidati.

Nel codice applicativo non viene calcolata una formula esplicita del tipo:

```text
alpha * dense_score + beta * sparse_score
```

Non sono presenti né pesi dense/sparse né un'implementazione locale di RRF. La combinazione del prefetch e la selezione dei candidati vengono delegate alla chiamata Qdrant.

### 6.4 Reranking ColBERT

Lo stesso `hybrid_search()` passa a Qdrant anche:

```python
query=query_vectors.colbert
```

Qdrant usa il multivector `colbert` della collection per rivalutare il pool dei candidati. Il comparatore configurato è `MAX_SIM`.

Il punteggio restituito viene salvato in:

```python
"_colbert_score": point.score
```

`_query_colbert()` calcola lo score del risultato migliore e imposta la soglia relativa a `top_score * SEMANTIC_RELEVANCE_RATIO`. Vengono mantenuti i punti con score maggiore o uguale a questa soglia.

Il flusso locale è quindi:

```text
dense prefetch + sparse prefetch
          |
          v
pool candidati Qdrant
          |
          v
ColBERT MaxSim
          |
          v
_colbert_score
          |
          v
filtro threshold
```

La cosiddetta fusion dense/sparse/ColBERT non è una somma manuale dei tre score. Dense e sparse contribuiscono al pool; ColBERT produce lo score usato per il ranking del risultato.

---

## 7. `retrieve_with_sources()`: retrieval di una domanda

Implementazione: [src/core/rag/retrieval.py](esperto/src/core/rag/retrieval.py), metodo `retrieve_with_sources()`.

### 7.1 Retrieval parallelo

Per tutte le query generate da `expand_queries()` viene eseguito:

```python
initial_results = await asyncio.gather(
    *(hybrid_search(...) for query_vectors in query_vectors)
)
```

### 7.2 Deduplicazione

`_keep_highest_scoring()` raggruppa per `chunk_id` e conserva il documento con il valore `_colbert_score` maggiore.

Se il `chunk_id` non è disponibile, usa il contenuto del documento come fallback.

Questo risolve duplicati provenienti da query diverse o da risultati ripetuti, ma non combina gli score: conserva solo il migliore.

### 7.3 Recupero aggiuntivo dello stesso file

Per ogni file individuato, il metodo esegue una nuova `hybrid_search()` filtrata con:

```python
FieldCondition(
    key="metadata.filename",
    match=MatchValue(value=filename),
)
```

Il limite è `4` chunk per file.

I chunk recuperati dallo stesso file entrano nello stesso insieme dei risultati e vengono ordinati con il loro punteggio ColBERT. Questa è un'espansione per file, non symbol graph e non una fusione di score.

### 7.4 Ordinamento prima del graph

`_document_sort_key()` ordina usando esclusivamente:

```python
-_colbert_score
```

Il chunk con il punteggio ColBERT maggiore viene prima, indipendentemente dalla sua provenienza.

### 7.5 Symbol graph expansion

Dal set dei chunk già ordinati viene chiamato:

```python
self._symbol_graph.related_chunks(chunk_ids, limit=SYMBOL_GRAPH_MAX_NEIGHBORS)
```

Implementazione runtime: [src/core/services/symbol_graph.py](esperto/src/core/services/symbol_graph.py).

Il graph opera in due fasi distinte:

1. **Raccolta dei candidati (BFS bidirezionale)**: parte dai simboli dei chunk
   recuperati, percorre gli edge `calls` e `inherits` in entrambe le direzioni
   (la direzione inversa è marcata `called_by:*`), evita i simboli già visitati
   e si ferma entro `SYMBOL_GRAPH_MAX_DEPTH` livelli. I chunk seed sono esclusi
   dai risultati. In questa fase non viene applicato alcun limite di 6.
2. **Ranking (Personalized PageRank diretto)**: sul sotto-grafo raccolto viene
   calcolato un PPR con personalizzazione uniforme sui seed. La propagazione
   usa solo l'adiacenza diretta originale (`source -> target`); gli archi
   sintetici `called_by:*` non entrano nella matrice di transizione. La dangling
   mass è redistribuita ai seed; l'iterazione si ferma per convergenza L1 o al
   massimo numero di iterazioni.

I rank dei simboli sono poi aggregati per chunk top-level, ordinati per score
decrescente con `chunk_id` come tie-breaker, e vengono restituiti al massimo
`SYMBOL_GRAPH_MAX_NEIGHBORS` chunk distinti.

I parametri attuali sono definiti in [src/core/shared/settings.py](esperto/src/core/shared/settings.py):

```text
SYMBOL_GRAPH_MAX_DEPTH = 3
SYMBOL_GRAPH_MAX_NEIGHBORS = 6
SYMBOL_GRAPH_PAGERANK_DAMPING = 0.4
SYMBOL_GRAPH_PAGERANK_ITERATIONS = 20
SYMBOL_GRAPH_PAGERANK_TOLERANCE = 1e-6
```

I chunk graph vengono poi recuperati direttamente da Qdrant tramite `get_documents_by_chunk_ids()`. Questa operazione usa gli ID, non dense/sparse/ColBERT.

A questi documenti vengono aggiunti:

```python
_graph_relation = "..."
```

Il valore della relazione è il percorso BFS più breve verso il simbolo, ad esempio:

```text
calls->called_by:calls
```

### 7.6 Budget del contesto

I documenti diretti e graph vengono concatenati, ordinati tramite `_document_sort_key()` e limitati da `_limit_context_documents()`.

Il budget comprende:

```text
RAG_CONTEXT_MAX_SOURCES = 35
RAG_CONTEXT_MAX_CHARS = 60_000
```

Il codice ordina i documenti per score ColBERT e li aggiunge finché non supera il numero massimo di fonti o il numero massimo di caratteri. Se un documento singolo è troppo grande dopo che almeno un documento è stato scelto, viene saltato.

### 7.7 Sources e contesto

Per ogni documento selezionato viene costruito un dizionario con:

```text
filename
service
chunk_type
symbol_name
start_line
end_line
chunk_id
score
graph_relation
content
```

Il contesto passato al modello è una stringa:

```text
content_1

---

content_2

---

content_3
```

---

## 8. `retrieve_many_with_sources()`: fusione delle domande del planner

Implementazione: [src/core/rag/retrieval.py](esperto/src/core/rag/retrieval.py), metodo `retrieve_many_with_sources()`.

Questo è il percorso usato normalmente da `stream_expert_response()`.

### 8.1 Costruzione delle domande di retrieval

Il metodo costruisce:

```text
original_question
+ sotto-domande diverse dalla domanda originale
```

Il totale viene limitato alle prime quattro domande.

### 8.2 Retrieval indipendente per domanda

Per ogni domanda viene invocato `retrieve_with_sources()` in parallelo:

```python
results = await asyncio.gather(
    *(self.retrieve_with_sources(question) for question in retrieval_questions)
)
```

Ogni domanda completa quindi il proprio percorso:

```text
hybrid search
-> deduplica
-> same-file expansion
-> symbol graph expansion
-> context budget locale
```

### 8.3 Fusione dei sources

I sources dei vari retrieval vengono accorpati in `sources_by_key`.

La chiave è:

```python
(filename, chunk_id)
```

Per la stessa coppia viene mantenuto il primo source incontrato con `setdefault()`. Non viene fatta una media o una somma degli score ottenuti dalle diverse sotto-domande.

Questa è la fusione multi-query applicativa:

```text
sources query 1
      + sources query 2
      + sources query 3
      + sources query 4
              |
              v
 deduplica per (filename, chunk_id)
```

### 8.4 Reranking ColBERT globale

Dopo la fusione, il metodo raccoglie tutti i `candidate_chunk_ids` e codifica nuovamente solo la domanda originale:

```python
original_vectors = await asyncio.to_thread(
    encode_queries,
    [original_question],
)
```

Poi chiama:

```python
reranked_documents = await rerank_chunk_ids(
    self._client,
    original_vectors[0],
    candidate_chunk_ids,
)
```

Implementazione di `rerank_chunk_ids()`: [src/core/services/qdrant.py](esperto/src/core/services/qdrant.py).

Questa query:

- non esegue il prefetch dense;
- non esegue il prefetch sparse;
- filtra Qdrant sugli ID già candidati;
- usa solo la query ColBERT;
- ricalcola `_colbert_score` per i candidati rispetto alla domanda originale.

Il risultato viene trasformato in:

```python
rerank_scores[chunk_id] = colbert_score
```

### 8.5 Filtro e ordinamento finale

Se esiste almeno uno score ColBERT globale, i sources vengono prima filtrati ai
soli candidati che superano la soglia relativa: chi non compare nel reranking
(sotto soglia) viene eliminato dal contesto. I sopravvissuti vengono ordinati
esclusivamente con:

```python
-final_score
```

Dove `final_score` è lo score ColBERT ricalcolato rispetto alla domanda
originale. Lo stesso valore è assegnato anche a `score`, così l'ordine
e il valore visualizzato coincidono.

Se il reranking globale non restituisce alcun candidato sopra soglia, il
contesto resta vuoto: nessun chunk debole viene passato all'LLM.

Infine viene applicato un unico budget condiviso di:

```text
RAG_CONTEXT_MAX_SOURCES
RAG_CONTEXT_MAX_CHARS
```

Questo è il contesto che arriva al prompt finale.

---

## 9. Dove avviene realmente la fusion

È utile separare quattro concetti che nel linguaggio comune vengono chiamati tutti "fusion".

### 9.1 Fusion dense/sparse

Avviene nella richiesta Qdrant di `hybrid_search()`:

```text
dense prefetch
      +
sparse prefetch
      |
      v
pool di candidati
```

Nel codice Python non c'è una formula esplicita di somma o media degli score dense e sparse.

### 9.2 Ranking ColBERT

Avviene in `_query_colbert()` per `hybrid_search()` e in `rerank_chunk_ids()` per il reranking globale.

ColBERT riceve il pool già candidato e produce `_colbert_score`.

### 9.3 Fusion tra sotto-domande

Avviene in `retrieve_many_with_sources()`:

```text
risultati domanda originale
+ risultati sotto-domande
-> deduplica per (filename, chunk_id)
-> reranking ColBERT rispetto alla domanda originale
-> eliminazione dei candidati sotto soglia relativa
```

Non vengono sommati i punteggi delle sotto-domande.

### 9.4 Fusion con symbol graph

Non è una fusion numerica.

Il graph raccoglie i candidati con BFS bidirezionale fino a 3 hop e li ordina
con Personalized PageRank diretto:

```text
chunk recuperati (seed)
      |
      v
BFS bidirezionale su calls/inherits (max 3 hop)
      |
      v
PPR diretto (source -> target) sul sotto-grafo
      |
      v
top 6 chunk correlati recuperati per ID
```

I punteggi PPR selezionano solo quali chunk graph entrare nel pool: nel percorso
`retrieve_many_with_sources()` anche questi chunk vengono sottoposti al
reranking globale ColBERT come gli altri candidati.


## 10. Prompt grounding e generazione

Implementazione: [src/core/rag/prompts.py](esperto/src/core/rag/prompts.py).

Dopo il retrieval, `stream_expert_response()` chiama:

```python
prompt_value = build_prompt().invoke(
    {"context": context, "question": question}
)
```

Il prompt contiene un singolo messaggio utente, costruito nell'ordine:

```text
CONTESTO RECUPERATO DALLA CODEBASE:
<context>

ISTRUZIONI:
<regole grounding>

DOMANDA:
<domanda originale>

RISPOSTA TECNICA:
```

Regole principali imposte al modello:

- usare esclusivamente il codice presente nel contesto;
- dichiarare di non sapere se la risposta non è deducibile;
- non inventare librerie, simboli o variabili;
- mostrare Prima/Dopo quando propone modifiche;
- citare file e funzione o linea quando disponibili;
- evitare contenuto generico o ripetitivo;
- rispondere in modo conciso.

Il contesto recuperato è quindi la sola evidenza fornita al modello finale.

### Debug del prompt

Se è presente `PROMPT_DEBUG_FILE`, `_save_last_prompt()` salva i messaggi renderizzati in un file JSON temporaneo e poi lo sostituisce atomicamente.

Nel compose attuale il file è montato come:

```text
/app/debug/last_prompt.json
```

---

## 11. Streaming della risposta

`_create_response_llm()` crea un `ChatOllama` per la risposta finale con:

```text
model = LLM_MODEL
base_url = OLLAMA_URL
temperature = TEMPERATURE
reasoning = use_reasoning
```

Il metodo `stream_expert_response()` emette prima le fonti:

```json
{"type":"sources","data":{"sources":[...],"subquestions":[...]}}
```

Poi itera su `astream(prompt_value)`.

Se il modello restituisce reasoning:

```json
{"type":"reasoning","data":"..."}
```

Se restituisce contenuto della risposta:

```json
{"type":"answer","data":"..."}
```

Il client deve quindi trattare lo stream come una sequenza di righe JSON, non come un unico JSON finale.

---

## 12. Health check e stati osservabili

### 12.1 Health dell'API RAG

Endpoint:

```text
GET /health :5000
```

Implementazione: [src/api.py](esperto/src/api.py), funzione `health_check()`.

Controlli:

1. esistenza del file `INDEX_STATE_DIR/index_complete`;
2. accessibilità della collection Qdrant;
3. numero di punti nella collection.

Risultati possibili:

```text
not_indexed -> manca il flag
indexing    -> collection vuota
ok          -> flag presente e punti Qdrant > 0
503         -> errore di accesso o controllo
```

### 12.2 Health dell'ingestion service

Endpoint:

```text
GET /health :5001
```

Restituisce semplicemente:

```json
{"status":"ok"}
```

Questo indica che il servizio ingestion risponde, non che un indice sia pronto.

### 12.3 Stato del job ingestion

Endpoint:

```text
GET /status :5001
```

Restituisce lo stato persistito del job. `ready` indica che il subprocess indicizzatore è terminato con successo.

---

## 13. Configurazione che controlla il comportamento

La configurazione viene caricata e validata in [config.py](esperto/config.py).

Parametri rilevanti:

```text
OLLAMA_URL                  endpoint Ollama
LLM_MODEL                   modello planner/risposta
TEMPERATURE                temperatura risposta finale
QDRANT_HOST                host Qdrant
QDRANT_PORT                porta Qdrant
QDRANT_COLLECTION          collection usata
K                          limite risultati primari
PREFETCH_LIMIT             candidati per dense e sparse
SEMANTIC_RELEVANCE_RATIO     rapporto k della soglia ColBERT rispetto al top score
EMBEDDING_MAX_LENGTH       lunghezza massima BGE-M3
QDRANT_QUERY_TIMEOUT       timeout query Qdrant
LLM_INFERENCE_TIMEOUT      timeout risposta LLM
BASE_DIR                   radice sorgenti indicizzati
INDEX_TARGET_PATHS         target discovery
SUPPORTED_EXTENSIONS       estensioni indicizzate
SKIP_DIRS                  directory escluse
BATCH_SIZE                 chunk per batch di upsert
INDEX_STATE_DIR            stato persistente
```

Costanti interne di retrieval in [src/core/shared/settings.py](esperto/src/core/shared/settings.py):

```text
DENSE_MODEL_NAME = BAAI/bge-m3
DENSE_DIMENSION = 1024
SYMBOL_GRAPH_MAX_NEIGHBORS = 6
SYMBOL_GRAPH_MAX_DEPTH = 3
SYMBOL_GRAPH_PAGERANK_DAMPING = 0.4
SYMBOL_GRAPH_PAGERANK_ITERATIONS = 20
SYMBOL_GRAPH_PAGERANK_TOLERANCE = 1e-6
RAG_CONTEXT_MAX_CHARS = 60_000
RAG_CONTEXT_MAX_SOURCES = 35
INDEX_CHUNK_SIZE = 1200
INDEX_CHUNK_OVERLAP = 150
```

---

## 14. Riassunto end-to-end

### 14.1 Primo caricamento

```text
client
  -> POST /upload
  -> validate_zip
  -> create queued job
  -> extract staging directory
  -> activate current symlink
  -> start python -m src.indicizzatore
  -> discover files
  -> chunk files
  -> build symbol graph
  -> encode dense/sparse/ColBERT
  -> upsert Qdrant
  -> save index state
  -> job ready
```

### 14.2 Domanda senza planner

```text
client
  -> POST /ask(use_planner=false)
  -> validate message
  -> [original question]
  -> encode dense/sparse/ColBERT
  -> Qdrant dense+sparse prefetch
  -> Qdrant ColBERT reranking
  -> deduplicate chunks
  -> same-file expansion
  -> symbol graph expansion (BFS 3 hop + PPR)
  -> context budget
  -> sources event
  -> grounded prompt
  -> Ollama streaming
  -> answer events
```

### 14.3 Domanda con planner

```text
client
  -> POST /ask(use_planner=true)
  -> planner LLM
  -> original question + max 3 subquestions
  -> retrieval parallelo per domanda
  -> deduplica sources per (filename, chunk_id)
  -> reranking globale ColBERT sulla domanda originale
  -> eliminazione dei candidati sotto la soglia relativa
  -> budget unico
  -> sources event
  -> grounded prompt
  -> risposta Ollama in streaming
```

### 14.4 Punto chiave sulla fusion

```text
Dense + Sparse = pool di candidati Qdrant
ColBERT         = score semantico dei candidati
Multi-query     = unione e deduplica dei chunk
Symbol graph    = candidati via BFS 3-hop, selezione via PPR diretto
Final ranking   = ColBERT sulla domanda originale, con filtro a soglia relativa
```

La fusione attuale è quindi una pipeline di **candidate retrieval, reranking,
deduplica, filtro di rilevanza ed espansione strutturale**, non una formula
unica che somma dense, sparse, ColBERT e graph score.
