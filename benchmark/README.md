# Benchmark RAG Parser

Benchmark esterno e isolato per misurare il sistema esperto sulla codebase del parser (`backend/` e `frontend/`). Non importa né modifica il codice di `esperto/`.

## Cosa produce

Ogni run crea due output con lo stesso timestamp:

- `results/<timestamp>.json`: metriche statiche, stato health e metriche runtime aggregate.
- `answers/<timestamp>.jsonl`: 50 coppie domanda-risposta per valutazione umana, una per riga.

Le metriche statiche sono stime indipendenti calcolate sul filesystem. Le metriche runtime sono osservazioni reali ottenute dagli endpoint pubblici `/health` e `/ask`.

## Prerequisiti

- Python 3.10 o superiore.
- Per la parte runtime, servizio esperto disponibile su `http://localhost:5000` oppure URL configurato.
- Nessuna dipendenza Python aggiuntiva.

## Uso

```bash
python3 benchmark/runner.py
```

Solo metriche statiche, senza chiamare il servizio:

```bash
python3 benchmark/runner.py --static-only
```

URL diverso:

```bash
python3 benchmark/runner.py --esperto-url http://localhost:5000
```

oppure:

```bash
ESPERTO_URL=http://localhost:5000 python3 benchmark/runner.py
```

## Configurazione

`config.json` controlla target di scansione, estensioni, directory escluse, timeout, dimensione massima dell'output, `use_planner` e `use_reasoning`. `queries.json` contiene esattamente 50 domande con `id`, `area` e `question`.

Il runner esegue una sola richiesta per domanda, senza retry automatici. Gli errori sono troncati e registrati per singola domanda, senza interrompere la run.

## Formato valutazione umana

Ogni riga di `answers/<timestamp>.jsonl` contiene:

- `id`, `area`, `question`
- `answer`
- `sources` e `sources_count`
- `latency_seconds`, `status`, `error`
- `timestamp`

Valori `status`:

- `ok`: risposta completa senza errori di stream.
- `partial`: risposta presente o stream incompleto/non valido.
- `failed`: richiesta non riuscita o assenza di risposta.

## Test

```bash
python3 -m unittest discover -s benchmark/tests
```

## Limiti dichiarati

- `estimated_chunks` approssima il chunking documentato: AST per Python e fallback testuale da 1200 caratteri con overlap 150.
- `ast_nodes` e `ast_edges_estimate` sono stime statiche, non una copia del symbol graph interno di `esperto`.
- Il benchmark non accede a Qdrant e non ricostruisce l'indice.
