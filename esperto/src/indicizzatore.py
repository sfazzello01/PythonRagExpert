"""Compatibility entry point for the Esperto indexing command."""

import logging

from .core.indexing.pipeline import build_index

__all__ = ["build_index"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_index()