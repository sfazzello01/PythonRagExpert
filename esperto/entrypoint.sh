#!/bin/sh

set -e

# L'indicizzazione NON parte piu' all'avvio: viene avviata esplicitamente
# dal servizio di ingestion (porta 5001) dopo l'upload di uno ZIP.

exec "$@"
