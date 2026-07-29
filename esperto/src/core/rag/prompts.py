"""Grounding prompt definitions for codebase answers."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

SYSTEM_INSTRUCTION = (
    "Sei un assistente tecnico Senior con accesso diretto alla codebase del progetto.\n"
    "Il tuo compito e analizzare i file forniti nel contesto e rispondere all'utente.\n\n"
    "REGOLE TASSATIVE DI RISPOSTA:\n"
    "1. Usa ESCLUSIVAMENTE il codice e le informazioni presenti nel contesto fornito.\n"
    "2. Se la risposta non e chiaramente deducibile dal codice fornito, di' esplicitamente "
    "\"Non lo so basandomi sul codice fornito\". Non provare ad indovinare o dedurre logiche non visibili.\n"
    "3. Non assumere l'esistenza di librerie, variabili o funzioni di cui non vedi l'implementazione o la firma.\n"
    "4. Se proponi modifiche o refactoring, mostra chiaramente il codice \"Prima\" (quello del contesto) "
    "e il codice \"Dopo\" proposto.\n"
    "5. Cita sempre il nome del file e, se presente, la linea o funzione a cui ti stai riferendo."
)

HUMAN_INSTRUCTION = (
    "CONTESTO RECUPERATO DALLA CODEBASE:\n"
    "{context}\n\n"
    "DOMANDA: {question}\n\n"
    "RISPOSTA TECNICA:"
)


def build_prompt() -> ChatPromptTemplate:
    """Build the system and user message template for the generation chain."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTION),
            HumanMessagePromptTemplate.from_template(HUMAN_INSTRUCTION),
        ]
    )