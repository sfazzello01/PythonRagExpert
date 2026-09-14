# Benchmark RAG Parser

Benchmark esterno e isolato per misurare il sistema esperto tramite la sua API pubblica. Non scansiona il filesystem, non importa né modifica il codice di `esperto/`.

## Cosa produce

Ogni run crea due output con lo stesso timestamp:

- `results/<timestamp>.json`: stato health, metriche runtime e metriche retrieval aggregate.
- `answers/<timestamp>.jsonl`: 50 coppie domanda-risposta per valutazione umana, una per riga.

Le metriche runtime sono osservazioni reali ottenute dagli endpoint pubblici `/health` e `/ask`.

## Prerequisiti

- Python 3.10 o superiore.
- Per la parte runtime, servizio esperto disponibile su `http://localhost:5000` oppure URL configurato.
- Nessuna dipendenza Python aggiuntiva.

## Uso

```bash
python3 benchmark/runner.py
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

`config.json` controlla endpoint, timeout, dimensione massima dell'output, `use_planner` e `use_reasoning`. `queries.json` contiene esattamente 50 domande con `id`, `area`, `question` e `relevant_sources`.

Il runner esegue una sola richiesta per domanda, senza retry automatici. Gli errori sono troncati e registrati per singola domanda, senza interrompere la run.

## Formato valutazione umana

Ogni riga di `answers/<timestamp>.jsonl` contiene:

- `id`, `area`, `question`
- `answer`
- `sources` e `sources_count`
- `relevant_sources`
- `retrieval_metrics`
- `latency_seconds`, `status`, `error`
- `timestamp`

Le metriche retrieval usano le prime cinque fonti uniche restituite:

- `precision_at_5`: fonti attese recuperate divise per fonti restituite.
- `recall_at_5`: fonti attese recuperate divise per fonti attese.
- `hit_at_5`: `1` se almeno una fonte attesa è nelle prime cinque, altrimenti `0`.
- `mrr`: inverso della posizione della prima fonte attesa recuperata.

I percorsi sono normalizzati per ignorare il prefisso storico `labnew/`. Il report include media, mediana, numero di domande e aggregazione per area.

Valori `status`:

- `ok`: risposta completa senza errori di stream.
- `partial`: risposta presente o stream incompleto/non valido.
- `failed`: richiesta non riuscita o assenza di risposta.

## Test

```bash
python3 -m unittest discover -s benchmark/tests
```

## Limiti dichiarati

- Le `relevant_sources` sono ground truth curate manualmente e valutano il retrieval, non la correttezza semantica della risposta.
- Il benchmark non accede a Qdrant e non ricostruisce l'indice.
- L'esecuzione richiede il servizio esperto attivo e invia una sola richiesta per ciascuna delle 50 domande.
