"""Funzioni per recuperare i dati dal database."""

import os

import mariadb  # type: ignore
from dotenv import load_dotenv

load_dotenv()


class DatabaseRetriever:
    def _connect(self) -> mariadb.Connection:
        """
        Crea e restituisce una nuova connessione al database MariaDB
        usando le variabili d'ambiente per la configurazione.

        returns:
            mariadb.Connection: una connessione al database
        """
        return mariadb.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
        )

    def _fetch_one(self, table, url) -> tuple | None:
        """
        Recupera un singolo record da una tabella specifica in base all'URL.

        args:
            table (str): nome della tabella da cui recuperare i dati
            url (str): URL della risorsa web da cercare

        returns:
            tuple | None: il record recuperato o None se non trovato
        """
        conn = None
        cursor = None

        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} WHERE url = ?", (url,))
            return cursor.fetchone()
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def get_web_resource(self, url) -> tuple | None:
        """
        Recupera una web resource dal database in base all'URL.

        args:
            url (str): URL della risorsa web da cercare

        returns:
            tuple | None: la risorsa web recuperata o None se non trovata
        """
        return self._fetch_one("web_resources", url)

    def get_gold_standard(self, url) -> tuple | None:
        """
        Recupera un gold standard dal database in base all'URL.

        args:
            url (str): URL del gold standard da cercare

        returns:
            tuple | None: il gold standard recuperato o None se non trovato
        """
        return self._fetch_one("gold_standard", url)

    def get_parsed_text(self, url) -> tuple | None:
        """
        Recupera un parsed text dal database in base all'URL.

        args:
            url (str): URL del parsed text da cercare

        returns:
            tuple | None: il parsed text recuperato o None se non trovato
        """
        return self._fetch_one("parsed_text", url)

    def get_evaluation(self, url) -> tuple | None:
        """
        Recupera una valutazione dal database in base all'URL.

        args:
            url (str): URL della valutazione da cercare

        returns:
            tuple | None: la valutazione recuperata o None se non trovata
        """
        return self._fetch_one("evaluation", url)

    def get_llm_eval(self, url) -> tuple | None:
        """
        Recupera una valutazione LLM dal database in base all'URL.

        args:
            url (str): URL della valutazione LLM da cercare

        returns:
            tuple | None: la valutazione LLM recuperata o None se non trovata
        """
        return self._fetch_one("llm_eval", url)
