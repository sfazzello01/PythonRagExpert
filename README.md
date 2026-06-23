# Progetto Laboratiorio 2026

## Attuale stato del progetto

- **Parser/scraper**: funzionante. Da ottimizzare lo scraping e la formattazione.
    - TODO: aggiungere 5 gs a dominio
    - TODO: sistemare i gs di trading economics
    - TODO: controllare e migliorare gli schemi degli scraper

- **Database**: ancora da testare meglio, ma funzioni principali implementate.

- **LLM as a judge**: implementate le funzioni per parlare con OLLAMA.

- **Log**: funziona correttamente, affinare i messaggi di errore e info.
    - TODO: aggiungere i log per i nuovi servizi

- **Server API**: completato, comunque ancora da testare meglio

- **Frontend**: da testare e sistemare meglio

- **Docker / Docker Compose**: tutto corretto per il funzionamento attuale

## Esecuzione del codice

Prima di lanciare il progetto assicurarsi di avere le porte necessarie libere e i vecchi container chiusi.
Per essere sicuri si possono lanciare i comandi:
```bash
sudo systemctl stop mariadb
sudo systemctl stop ollama
docker compose down
```

Se non si ha una scheda NVIDIA vanno commentate le linee da 56 a 62 nel file `docker-compose.yaml`

A riga 64 sempre del file `docker-compose.yaml` bisogna selezionare il modello che si vuole scaricare e usare

per far partire il progetto scrivere:
```bash
docker compose up --build
```

dopo l'avvio della build aspettare che come ultima riga sul terminale appaia    
`backend   | INFO:     Uvicorn running on http://0.0.0.0:8003 (Press CTRL+C to quit)`

per testare il `llm_judge` usare questo comando in un altro terminale: 
```
docker exec -it backend python -m src.llm_judge.test_llm
```
l'LLM può essere attivato modificando il parametro ```NO_LLM``` a ```False```

per testare il `database` usare invece questo comando: 
```
docker exec -it backend python -m src.database.test_db
```