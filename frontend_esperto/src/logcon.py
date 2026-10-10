"""Configurazione del logging del frontend: file di log e console."""

import logging

from config import LOGS_DIR, LOGS_FILE


class LoggerConfig:
    """Impostazioni condivise dai logger del frontend; alla definizione crea la cartella dei log."""

    LOG_DIR = LOGS_DIR
    LOG_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOGS_FILE
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Restituisce un logger che scrive su file e console, configurandolo al primo uso.

    Args:
        name: Nome del logger.

    Returns:
        Il logger configurato.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LoggerConfig.LOG_LEVEL)
    logger.propagate = False

    file_handler = logging.FileHandler(LoggerConfig.LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(LoggerConfig.LOG_LEVEL)
    file_handler.setFormatter(LoggerConfig.LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LoggerConfig.LOG_LEVEL)
    console_handler.setFormatter(LoggerConfig.LOG_FORMAT)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
