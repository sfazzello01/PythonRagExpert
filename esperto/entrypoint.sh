#!/bin/sh

set -e

python -m src.indicizzatore

exec "$@"
