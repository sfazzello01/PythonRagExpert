"""Grounding prompt definitions for codebase answers."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

# Singolo messaggio utente: alcuni modelli Ollama gestiscono male i SystemMessage separati e
# restituiscono risposte vuote. La mappa, uguale per ogni richiesta, apre il prompt (Ollama ne
# riusa la cache); la domanda precede il contesto, cosi' il modello lo legge sapendo cosa cercare,
# ed e' ripetuta in fondo. Poche regole: un modello piccolo ne segue bene poche.
INSTRUCTION = (
    "{repo_map}"
    "DOMANDA A CUI RISPONDERE: {question}\n\n"
    "CONTESTO RECUPERATO DALLA CODEBASE:\n"
    "{context}\n\n"
    "ISTRUZIONI:\n"
    "Sei un assistente tecnico con accesso alla codebase del progetto. "
    "Rispondi alla domanda usando il contesto qui sopra.\n"
    "1. Usa solo il codice e le informazioni del contesto: non inventare file, funzioni o librerie.\n"
    "2. Se la risposta non è nel contesto, scrivi \"Non lo so basandomi sul codice fornito\".\n"
    "3. Cita il file e la funzione o classe a cui ti riferisci.\n"
    "4. Se proponi modifiche, mostra il codice \"Prima\" e \"Dopo\".\n"
    "5. Sii conciso: niente premesse né ripetizioni.\n\n"
    "DOMANDA: {question}\n\n"
    "RISPOSTA TECNICA:"
)


def build_prompt() -> ChatPromptTemplate:
    """Build the prompt template for the generation chain (single user message)."""
    return ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(INSTRUCTION),
        ]
    )