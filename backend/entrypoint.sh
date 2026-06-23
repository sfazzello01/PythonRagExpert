#!/bin/sh

set -e

# Initialize the database before starting the API.
python -m src.database.inizializzazione
python -m src.database.seed

exec "$@"