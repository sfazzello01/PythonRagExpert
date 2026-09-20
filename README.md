# Esperto RAG: Architettura Ibrida Avanzata per la Comprensione Semantica e Topologica del Codice Sorgente

**Documento di Tesi e Analisi Algoritmica Approfondita**

---

## Abstract

Il progetto **Esperto RAG** è stato sviluppato attraverso un'architettura ibrida all'avanguardia. Il sistema unisce l'inferenza generativa dei Large Language Models (LLM) con un infrastruttura di embedding multi-vettoriale (Denso, Sparso, ColBERT) e un motore di esplorazione topologica basato sulla costruzione di *Abstract Syntax Tree* (AST) e su algoritmi di teoria dei grafi, quali la *Breadth-First Search* (BFS) e il *Personalized PageRank* (PPR).


---

## Indice

1. [Capitolo 1: Introduzione e Formulazione del Problema](#capitolo-1-introduzione-e-formulazione-del-problema)
2. [Capitolo 2: Fondamenti Teorici e Modelli Algoritmici](#capitolo-2-fondamenti-teorici-e-modelli-algoritmici)
   - 2.1 Vettorializzazione Densa e Sparsa
   - 2.2 Late Interaction e ColBERT (MaxSim)
   - 2.3 Teoria dei Grafi nel Codice: BFS
   - 2.4 Personalized PageRank (PPR)
3. [Capitolo 3: Architettura del Sistema e Infrastruttura](#capitolo-3-architettura-del-sistema-e-infrastruttura)
4. [Capitolo 4: Flusso di Ingestion e Indicizzazione Strutturale (Pipeline Offline)](#capitolo-4-flusso-di-ingestion-e-indicizzazione-strutturale-pipeline-offline)
   - 4.1 Ricezione, Validazione e Gestione del Job
   - 4.2 Workspace Staging e Fingerprinting
   - 4.3 Discovery dei File
   - 4.4 Chunking Contestuale basato su AST
   - 4.5 Generazione del Symbol Graph
   - 4.6 Embedding Multi-Rappresentazione e Upsert su Qdrant
5. [Capitolo 5: Flusso di Interrogazione e Retrieval Avanzato (Pipeline Online)](#capitolo-5-flusso-di-interrogazione-e-retrieval-avanzato-pipeline-online)
   - 5.1 Il Planner LLM e la Generazione di Sotto-Domande
   - 5.2 Hybrid Search: Prefetching Parallelo Denso e Sparso
   - 5.3 Reranking ColBERT Locale
   - 5.4 Fusione e Deduplica Intra-File
   - 5.5 Espansione Topologica via Grafo (BFS + PPR)
   - 5.6 Global Reranking ColBERT e Filtro di Soglia Relativa
   - 5.7 Context Budgeting e Limitazione del Rumore
6. [Capitolo 6: Prompt Grounding e Generazione della Risposta](#capitolo-6-prompt-grounding-e-generazione-della-risposta)
7. [Capitolo 7: Analisi Critica: Il Modello "Candidate Funnel" vs "Linear Fusion"](#capitolo-7-analisi-critica-il-modello-candidate-funnel-vs-linear-fusion)
8. [Conclusioni](#conclusioni)

---

## Capitolo 1: Introduzione e Formulazione del Problema

Il paradigma RAG convenzionale si basa su due fasi elementari: il partizionamento del testo in blocchi di dimensione fissa (spesso sovrapposti) e la codifica di tali blocchi in uno spazio vettoriale continuo ad alta dimensionalità (Embedding Denso). Quando un utente pone una domanda, la metrica della distanza coseno tra il vettore della query e i vettori dei documenti determina quali frammenti recuperare e iniettare nel prompt del LLM.

Nell'ecosistema dello sviluppo software, questo approccio elementare fallisce per tre motivi fondamentali:
1. **La Cecità Topologica**: Se un blocco di codice Python invoca la funzione `calcola_tasse()`, il LLM deve disporre anche dell'implementazione di tale funzione per fornire una risposta accurata. Il RAG tradizionale non ha alcuna cognizione di dove si trovi `calcola_tasse()`, a meno che i due blocchi non condividano accidentalmente una prossimità semantica nel vettore denso, cosa che raramente accade per funzioni utility generiche.
2. **Il Problema della Diluizione Lessicale**: L'embedding denso "spalma" il significato di una parola sull'intero vettore. Variabili cruciali come `user_db_v2` vengono astratte nel concetto generale di "connessione database". Una ricerca che richiede esattamente `user_db_v2` fallirà spesso in favore di blocchi che parlano di database in termini discorsivi.
3. **Frammentazione Strutturale**: Il chunking a finestre fisse tronca cicli `for`, divide definizioni di classi e separa le istruzioni di importazione dalle funzioni che le utilizzano, rendendo il frammento orfano e incomprensibile per il modello generativo.

Esperto RAG affronta il primo problema estraendo un grafo delle dipendenze direttamente dalla sintassi, il secondo introducendo vettori sparsi (BM25-like) e modelli di *late interaction* (ColBERT), e il terzo sostituendo il chunking testuale con un partizionamento basato sull'Albero Sintattico Astratto (AST).

---

## Capitolo 2: Fondamenti Teorici e Modelli Algoritmici

Per comprendere l'architettura di Esperto RAG, è essenziale formalizzare i modelli matematici e gli algoritmi sottostanti.

### 2.1 Vettorializzazione Densa e Sparsa
*   **Vettori Densi**: Vettori a dimensionalità fissa (es. 1024 dimensioni) composti da numeri in virgola mobile continui. Mappano il concetto semantico in uno spazio latente. La similarità è tipicamente misurata tramite la distanza Coseno: 
    $$ \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} $$
*   **Vettori Sparsi**: Vettori con una dimensionalità pari alla grandezza del vocabolario (es. 100.000+ dimensioni), dove la quasi totalità degli elementi è zero. Assomigliano al modello TF-IDF o BM25. Misurano l'impatto lessicale esatto di una singola keyword. La similarità è calcolata tramite prodotto scalare, privilegiando i match sintattici secchi.

### 2.2 Late Interaction e ColBERT (MaxSim)
I modelli densi classici operano una *Early Interaction* (l'intero documento viene schiacciato in un solo vettore) o nessuna iterazione token-level. 
ColBERT (Contextualized Late Interaction over BERT) rappresenta un'evoluzione radicale. Invece di produrre un singolo vettore, il modello genera **un vettore di $d$ dimensioni per ogni singolo token** del testo.
Dati $Q$ (matrice dei vettori dei token della query) e $D$ (matrice dei vettori dei token del documento), la similarità viene calcolata tramite l'operatore **MaxSim**:
$$ \text{Score}(Q, D) = \sum_{i \in Q} \max_{j \in D} (Q_i \cdot D_j) $$
Questo significa che per ogni token della domanda, il sistema cerca il token più simile semanticamente all'interno del documento e ne somma i contributi. Questo processo ("Late Interaction") garantisce un livello di precisione granulare ineguagliabile nell'estrazione tecnica, ideale per il codice sorgente.

### 2.3 Teoria dei Grafi nel Codice: BFS
La *Breadth-First Search* (Ricerca in Ampiezza) è un algoritmo di visita dei grafi. Inizia da un nodo sorgente e ne visita tutti i vicini diretti prima di procedere al livello successivo di profondità (hop). In Esperto RAG, questo algoritmo esplora topologicamente le relazioni di codice: se si parte da una funzione `Seed`, la BFS a 3-hop individua tutte le funzioni da essa chiamate, le funzioni che chiamano quelle chiamate, e le classi da esse derivate, garantendo di catturare il cluster logico di esecuzione.

### 2.4 Personalized PageRank (PPR)
Il PageRank originale calcola l'importanza di un nodo in base alla probabilità asintotica che un *random walker* (navigatore casuale) vi giunga attraversando il grafo. Il **Personalized PageRank (PPR)** altera questa metrica, vincolando il random walker a "teletrasportarsi" (re-iniziare il suo cammino) non in un nodo casuale qualsiasi, ma esclusivamente in un sottoinsieme di nodi "Seed".
Matematicamente, si definisce:
$$ R = (1 - d) p + d M R $$
Dove:
*   $R$ è il vettore dei ranghi dei nodi.
*   $d$ è il fattore di smorzamento (Damping factor, es. $0.4$).
*   $p$ è il vettore di personalizzazione (non nullo solo per i chunk originariamente recuperati).
*   $M$ è la matrice di transizione probabilistica del sotto-grafo esplorato.

Questo algoritmo permette di rispondere alla domanda: *"Dato questo set di funzioni di partenza, quali sono le classi o i file globalmente più influenti ed essenziali per il loro funzionamento?"*.

---

## Capitolo 3: Architettura del Sistema e Infrastruttura

Esperto RAG non è un monolita, ma un ecosistema di microservizi dockerizzati. La struttura separa nettamente i processi asincroni di background (I/O intensivi) da quelli di interrogazione sincrona.

1.  **Container `esperto` (Porta 5000)**: Cuore pulsante online dell'interfaccia RAG. Espone l'API FastAPI, orchestra le sotto-query, la navigazione del grafo, interroga Qdrant e innesca lo stream LLM.
2.  **Container `esperto_indexer` (Porta 5001)**: Lavoratore asincrono per l'ingestion. Possiede la logica di estrazione ZIP, l'analisi sintattica AST e gestisce i subprocessi per l'indicizzazione vettoriale per non saturare l'event loop HTTP.
3.  **Container `qdrant` (Porta 6333)**: Database vettoriale di ultima generazione scritto in Rust. È configurato esplicitamente per accogliere payload *Ibridi* e Multi-vettoriali (Dense, Sparse e ColBERT) sulla stessa collection.
4.  **Container `ollama` (Porta 11434)**: Modello LLM inferenziale eseguito su acceleratore grafico (NVIDIA GPU). Per supportare codebase estese, adotta **Flash Attention** (un algoritmo di self-attention con complessità di memoria $O(N)$ invece di $O(N^2)$) e quantizzazione della cache KV.
5.  **Container `frontend_esperto` (Porta 8005)**: UI utente.

---

## Capitolo 4: Flusso di Ingestion e Indicizzazione Strutturale (Pipeline Offline)

Questo capitolo sviscera l'implementazione esatta che avviene all'arrivo di uno `.zip` via `POST /upload`.

### 4.1 Ricezione, Validazione e Gestione del Job
Il file non viene brutalmente decompresso. Il file `src/ingestion/zip_validation.py` esegue una suite di controlli: verifica la firma binaria ZIP, la dimensione totale (limite in MB), impedisce attacchi di *path traversal* (negando stringhe come `../`) e filtra i contenuti tramite un set rigoroso di `SUPPORTED_EXTENSIONS`.
Il servizio crea un job di indicizzazione, persistendone lo stato (`not_indexed`, `queued`, `indexing`, `ready`, `error`) nel file `index_jobs.json`. Restituisce immediatamente un codice HTTP `202 Accepted` al client.

### 4.2 Workspace Staging e Fingerprinting
Per garantire isolamento transazionale, i file validi vengono estratti in `WORKSPACE_ROOT/job-<id>`. Per passare alla nuova versione senza creare downtime (zero-downtime deployment per il RAG), il sistema ricorre alla modifica atomica del puntatore di sistema operativo `os.replace()`, aggiornando un link simbolico da `current` alla nuova cartella di staging.
Successivamente viene generato un **fingerprint** crittografico combinando l'identità dei file. Se l'indice Qdrant esiste e il fingerprint coincide, l'indicizzazione viene saltata con successo.

### 4.3 Discovery dei File
Il modulo `discovery.py` risolve i percorsi ed esclude iterativamente le cartelle non necessarie (es. `node_modules`, `venv` mappate in `SKIP_DIRS`). Assegna inoltre a ciascun file un tag `service` dedotto dal primo segmento del path, utile al LLM per orientarsi in architetture a microservizi.

### 4.4 Chunking Contestuale basato su AST
Il partizionamento è asimmetrico:
1.  **File non supportati (o con Syntax Error)**: Retrocedono a un partizionatore standard (1200 caratteri di dimensione, 150 di sovrapposizione).
2.  **L'algoritmo PythonASTContextualChunker**: Per i file Python (il cuore della codebase), il file viene parsato tramite `ast.parse()`. Vengono estratti iterativamente nodi come `ClassDef`, `FunctionDef` e `AsyncFunctionDef`.
    Crucialmente, il chunker costruisce stringhe composite. Un frammento non contiene solo la logica della funzione, ma viene iniettato con le dipendenze globali del file:
    ```text
    # FILE: src/db/connection.py
    # TIPO: FUNCTION
    --- IMPORTS DI CONTESTO ---
    import sqlalchemy
    from config import DB_URL
    --- CODICE SORGENTE ---
    def get_db_session():
        ...
    ```
    Questa iniezione è vitale: senza di essa, il LLM non saprebbe da quale libreria proviene un oggetto instanziato all'interno della funzione. A ciascun chunk viene assegnato un `chunk_id` deterministico derivato dal path e dal nome del simbolo.

### 4.5 Generazione del Symbol Graph
Mentre esplora l'AST, il sistema traccia meticolosamente le relazioni. I risultati vengono salvati atomicamente in `symbol_graph.json`.
*   **Definizione Nodi**: Contiene `id`, `kind` (module, class, function), `name`, `filename`, `start_line`, `end_line`, `chunk_id`.
*   **Definizione Archi (Edges)**: Contiene il nodo sorgente, il nodo target, il tipo di arco (`calls`, `inherits`, `imports` o `defines`) e la linea esatta in cui avviene la relazione. 

### 4.6 Embedding Multi-Rappresentazione e Upsert su Qdrant
L'istanza di embedding carica il modello **BGE-M3**. Ogni `Document` generato (sia esso da AST o da testo di fallback) viene encodato per ottenere la triade Densità, Sparsità e Multi-vettore ColBERT.
In `Qdrant` (configurato con un comparatore `MAX_SIM` per il multivector), i frammenti sono inseriti a lotti (Batch Upsert). L'identificativo all'interno di Qdrant è un UUID univocamente derivato dal `chunk_id` applicativo: questo garantisce che, in futuro, aggiornare un singolo file aggiornerà esattamente il record Qdrant senza duplicazioni ("idempotenza vettoriale").
A completamento, il job segna il flag `index_complete`.

---

## Capitolo 5: Flusso di Interrogazione e Retrieval Avanzato (Pipeline Online)

Quando un client esegue `POST /ask`, l'obiettivo del sistema è recuperare il sotto-insieme perfetto di codice per fornire un *Grounding Context* incontestabile. Questo flusso non implementa formule lineari banali, ma un'elaborata pipeline a imbuto (Funnel Pipeline).

### 5.1 Il Planner LLM e la Generazione di Sotto-Domande
La query originaria spesso è ambigua o troppo complessa per il singolo vettore.
Se il flag `use_planner=true` è impostato, l'LLM (`ChatOllama`) viene interpellato pre-retrieval con parametro `temperature=0` (nessuna creatività) e `reasoning=false`. Il prompt del Planner richiede di generare un array JSON con un **massimo di 4 sotto-domande** che spacchettino il problema (es. chiedendo l'implementazione del DB, l'API e i modelli dati). La risposta viene pulita da errori di parsing JSON e duplicati. Se il modello fallisce, si degrada graziosamente utilizzando solo la domanda originaria.

### 5.2 Hybrid Search: Prefetching Parallelo Denso e Sparso
Per la query originale e ogni sotto-domanda, il RAG orchestra thread concorrenti tramite `asyncio.gather`.
Ogni domanda viene codificata e passata a Qdrant tramite una procedura di `Prefetch`. La query al DB richiede esplicitamente:
1.  **Dense Prefetch**: Recupera i migliori candidati usando la metrica *Distance.COSINE*.
2.  **Sparse Prefetch**: Recupera i migliori candidati in parallelo tramite pesi lessicali.
Qdrant popola il "Pool dei candidati" unendo questi due set. Notare l'assenza di somme lineari: i due set sono fusi insieme senza pesi relativi.

### 5.3 Reranking ColBERT Locale
All'interno della stessa richiesta Qdrant, il *Pool dei candidati* ibrido viene sottoposto alla query ColBERT (multivettore). Qdrant applica l'operatore **MaxSim** per generare il `_colbert_score` per ciascun candidato. Il DB calcola il punteggio maggiore e applica una soglia (`top_score * SEMANTIC_RELEVANCE_RATIO`), rimuovendo i documenti che cadono al di sotto.

### 5.4 Fusione e Deduplica Intra-File
Con il ritorno dei risultati paralleli (generati dalle sotto-domande del planner), le liste si fondono. 
1.  **Deduplicazione**: Se due query hanno recuperato lo stesso frammento, esso viene deduplicato. Come score di riferimento viene mantenuto esclusivamente il `_colbert_score` più alto mai raggiunto da tale frammento in qualsiasi query.
2.  **Same-File Expansion**: Se una classe importante è stata recuperata, il LLM potrebbe aver bisogno delle funzioni limitrofe dello stesso file per comprenderne il contesto completo. Il sistema lancia una query Qdrant filtrata via `metadata.filename`, estraendo fino a 4 chunk addizionali per file e ordinandoli per score ColBERT.

### 5.5 Espansione Topologica via Grafo (BFS + PPR)
Questo passaggio rende il sistema consapevole dell'architettura del software.
Il set parziale dei candidati viene fornito come "Nodi Seed" al gestore del *Symbol Graph*.
1.  **Raccolta Candidati (BFS Bidirezionale)**: L'algoritmo procede per 3 hop (profondità massima `SYMBOL_GRAPH_MAX_DEPTH = 3`). Attraversa gli archi `calls` (chi chiama), genera al volo archi `called_by` (chi è chiamato da chi) e attraversa l'albero di ereditarietà (`inherits`). Questo crea un sotto-grafo locale.
2.  **Ranking Assoluto (PPR Diretto)**: Sul sotto-grafo, viene applicato l'algoritmo di Personalized PageRank. L'algoritmo re-inizializza il random walker (probabilità $p$) unicamente sui nodi "Seed", escludendoli poi dal risultato finale. Le iterazioni della matrice di transizione di Markov continuano fino a un massimo di 20 (`SYMBOL_GRAPH_PAGERANK_ITERATIONS`), smorzate da un damping factor del $0.4$, o finché la differenza L1 non scende sotto $1e-6$ (`SYMBOL_GRAPH_PAGERANK_TOLERANCE`).
3.  I top 6 risultati (`SYMBOL_GRAPH_MAX_NEIGHBORS = 6`) con il rank PPR più elevato divengono candidati "Topologici". Questo meccanismo recupera ad esempio una `BaseClass` essenziale anche se essa non conteneva alcuna keyword menzionata nella query dell'utente.

### 5.6 Global Reranking ColBERT e Filtro di Soglia Relativa
A questo punto della pipeline, il sistema possiede una cozzaglia eterogenea di frammenti recuperati tramite semantica (Denso), keyword (Sparso), sotto-query e matematica topologica (Grafo PPR).
Questa disomogeneità viene risolta dal **Reranking Globale**. 
Il sistema preleva tutti gli ID recuperati e *ri-codifica* esclusivamente la **Domanda Originaria dell'Utente**.
Qdrant viene interpellato nuovamente: la query forza la misurazione ColBERT (MaxSim) della domanda originaria su tutti questi candidati, producendo un ricalcolo globale univoco. 
I candidati vengono filtrati per l'ultima volta contro la soglia relativa (`SEMANTIC_RELEVANCE_RATIO`) rispetto al nuovo top-scorer, escludendo matematicamente ogni elemento rivelatosi inutile.

### 5.7 Context Budgeting e Limitazione del Rumore
I candidati sopravvissuti vengono ordinati esclusivamente tramite il punteggio `-final_score` (il punteggio ColBERT globale appena ricalcolato).
Viene applicato il budget limitato dalle costanti globali per evitare il collasso della *Context Window Attention* del LLM:
*   `RAG_CONTEXT_MAX_SOURCES = 35`
*   `RAG_CONTEXT_MAX_CHARS = 60_000`
Se l'aggiunta di un documento fa superare queste soglie, esso viene scartato dal prompt.

---

## Capitolo 6: Prompt Grounding e Generazione della Risposta

L'epilogo del processo non è una mera concatenazione, ma la costruzione di un Prompt Systemico vincolante, noto come **Strict Grounding**.

Il contesto finale viene iniettato nel prompt sotto l'intestazione:
`CONTESTO RECUPERATO DALLA CODEBASE:`
A seguire, le **Regole di Grounding** impongono al LLM un rigore assoluto:
1. Deve usare *esclusivamente* il codice presente nel contesto.
2. Deve dichiarare di non possedere informazioni (rifiuto) qualora i frammenti siano insufficienti.
3. È vietato "allucinare" o indovinare l'esistenza di librerie, import o nomi di variabili non presenti.
4. Ogni codice proposto in refactoring deve adottare la visualizzazione "Prima / Dopo".
5. Deve citare pedissequamente i `filename` estratti nei metadati.

L'output del modello LLM (Ollama) è processato in streaming. Il client riceve i pacchetti sotto forma di Server-Sent Events (SSE) decodificati in JSON multivariati:
*   `{"type":"sources", "data": [...]}`: Trasmesso per primo, contiene i riferimenti ai file.
*   `{"type":"reasoning", "data": "..."}`: Trasmesso progressivamente se il LLM implementa una catena di pensieri.
*   `{"type":"answer", "data": "..."}`: Lo stream di testo principale e finale, processato man mano che arriva il token, assicurando un *Time to First Byte* (TTFB) minimale.

In background, l'API RAG offre endpoint di *Health Checking* (`GET /health` su porta 5000), verificando atomicamente il file flag `index_complete`, sondando il demone di rete di Qdrant e confermando la presenza dei vettori inseriti, per garantire elevata robustezza e reattività.

---

## Capitolo 7: Analisi Critica: Il Modello "Candidate Funnel" vs "Linear Fusion"

In letteratura tecnica e accademica, il termine "Hybrid Fusion" viene spesso confuso con la *Reciprocal Rank Fusion* (RRF) o con interpolazioni lineari del tipo:
$$ \text{Score}_{Finale} = \alpha \cdot \text{Score}_{denso} + \beta \cdot \text{Score}_{sparso} $$
Questo sistema di combinazione si rivela metricamente scorretto poiché somma variabili casuali provenienti da distribuzioni probabilistiche ortogonali (l'iper-sfera Coseno per la densità concettuale, l'intersezione lessicale per la sparsità TF-IDF, la Markov Chain probabilistica per il PageRank).

Esperto RAG scarta intenzionalmente questo approccio lineare in favore del **Modello a Imbuto Categoriale (Candidate Funnel)**.
1.  **La Fase di Selezione (Reti a strascico asimmetriche)**: Il Dense Prefetch, lo Sparse Prefetch e la BFS del Graph non sono utilizzati per generare punteggi definitivi. Il loro scopo è puramente topologico: recuperare *tutto* il materiale potenzialmente rilevante. La loro eterogeneità garantisce che nessuna keyword esatta, nessun concetto ampio e nessuna dipendenza di classe vitale vada persa.
2.  **La Fase del Giudizio (Authoritative Ranking)**: Nessuno degli "score" di recupero originari (Distanza Coseno, Rank BFS, BM25) viene addizionato. Invece, l'intero pool di candidati passa per il collo dell'imbuto dove risiede un unico giudice matematico omogeneo: **ColBERT MaxSim**. ColBERT, misurando la vicinanza a livello di token incrociato, è l'unico algoritmo autorizzato a decretare la pertinenza reale del codice rispetto alla domanda originale, emettendo un ranking globale coerente e netto.

---

## Conclusioni

Il progetto *Esperto RAG* rappresenta una netta evoluzione nell'applicazione delle Intelligenze Artificiali Generative al dominio dell'Ingegneria del Software. Discostandosi dagli algoritmi tradizionali sviluppati per il testo naturale, il sistema implementa un parsing sintattico profondo, la decostruzione topologica dei sorgenti e una pipeline matematica multi-stadio per il recupero delle informazioni (Candidate Funneling).
Sfruttando un'infrastruttura di microservizi dockerizzata con supporto hardware GPU (CUDA/Flash Attention), il sistema è in grado di rintracciare e contestualizzare porzioni di codice partendo non solo dal significato astratto del prompt utente, ma ancorandosi solidamente all'effettiva architettura delle dipendenze del software e fornendo al LLM un *Grounding* assoluto contro le allucinazioni tecniche.
