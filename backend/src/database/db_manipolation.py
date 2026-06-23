"""classe per la manipolazione del database."""

import os
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import mariadb  # type: ignore
from dotenv import load_dotenv

load_dotenv()


class DatabaseManipulator:
    def _connect(self) -> mariadb.Connection:
        """
        Stabilisce una connessione al database MariaDB utilizzando le variabili d'ambiente per i parametri di connessione.

        Returns:
            mariadb.Connection: Un oggetto di connessione al database MariaDB.
        """
        return mariadb.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
        )

    @staticmethod
    def _now() -> datetime:
        """
        Restituisce la data e ora corrente in formato datetime, con timezone Europe/Rome.

        Returns:
            datetime: La data e ora corrente con timezone Europe/Rome.
        """
        return datetime.now(ZoneInfo("Europe/Rome"))

    @staticmethod
    def _esiste(cursor, table, url) -> bool:
        """
        Controlla se un record con un URL specifico esiste già in una tabella del database

        args:
            cursor (mariadb.Cursor): Il cursore del database da utilizzare per la query
            table (str): Il nome della tabella in cui cercare
            url (str): L'URL da cercare nella tabella

        returns:
            bool: True se il record esiste, False altrimenti
        """
        cursor.execute(f"SELECT url FROM {table} WHERE url = ?", (url,))
        return cursor.fetchone() is not None

    @contextmanager
    def _sessione(self) -> mariadb.Cursor:
        """
        Contesto per gestire la connessione al database e il commit automatico.
        Garantisce che la connessione venga chiusa correttamente anche in caso di errori.

        Yields:
            mariadb.Cursor: un cursore per eseguire le operazioni sul database
        """
        conn = None
        cursor = None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def _upsert(
        self,
        table,
        url,
        insert_sql,
        insert_values,
        update_sql,
        update_values,
        insert_message,
        update_message,
    ) -> None:
        """
        Esegue un'operazione di upsert (insert o update) su una tabella del database.

        args:
            table (str): nome della tabella su cui operare
            url (str): URL che identifica univocamente la risorsa
            insert_sql (str): query SQL per l'inserimento
            insert_values (tuple): valori da inserire nella query di inserimento
            update_sql (str): query SQL per l'aggiornamento
            update_values (tuple): valori da inserire nella query di aggiornamento
            insert_message (str): messaggio da stampare in caso di inserimento
            update_message (str): messaggio da stampare in caso di aggiornamento
        """
        with self._sessione() as cursor:
            if self._esiste(cursor, table, url):
                cursor.execute(update_sql, update_values)
                print(update_message.format(url=url))
            else:
                cursor.execute(insert_sql, insert_values)
                print(insert_message.format(url=url))

    def _elimina(self, table, url, message) -> None:
        """
        Elimina un record da una tabella del database.

        args:
            table (str): nome della tabella da cui eliminare il record
            url (str): URL che identifica univocamente la risorsa
            message (str): messaggio da stampare dopo l'eliminazione
        """
        with self._sessione() as cursor:
            cursor.execute(f"DELETE FROM {table} WHERE url = ?", (url,))
            print(message.format(url=url))

    def insert_web_resource(self, url, domain, title, html_text) -> None:
        """
        Inserisce o aggiorna una web resource nel database.

        args:
            url (str): URL della risorsa web
            domain (str): dominio della risorsa
            title (str): titolo della risorsa
            html_text (str): testo HTML della risorsa
        """
        created_at = self._now()
        self._upsert(
            table="web_resources",
            url=url,
            insert_sql="""
                INSERT INTO web_resources (url, domain, title, html_text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
            insert_values=(url, domain, title, html_text, created_at),
            update_sql="""
                UPDATE web_resources
                SET domain = ?, title = ?, html_text = ?, created_at = ?
                WHERE url = ?
            """,
            update_values=(domain, title, html_text, created_at, url),
            insert_message="Web resource '{url}' inserita con successo!",
            update_message="Web resource '{url}' aggiornata con successo!",
        )

    def insert_gold_standard(self, url, gold_text) -> None:
        """
        Inserisce o aggiorna un gold standard nel database.

        args:
            url (str): URL della risorsa web a cui si riferisce il gold standard
            gold_text (str): testo del gold standard
        """
        print(f"Inserimento/aggiornamento gold standard per URL: {url}")
        created_at = self._now()
        self._upsert(
            table="gold_standard",
            url=url,
            insert_sql="""
                INSERT INTO gold_standard (url, gold_text, created_at)
                VALUES (?, ?, ?)
            """,
            insert_values=(url, gold_text, created_at),
            update_sql="""
                UPDATE gold_standard
                SET gold_text = ?, created_at = ?
                WHERE url = ?
            """,
            update_values=(gold_text, created_at, url),
            insert_message="Gold standard per '{url}' inserito con successo!",
            update_message="Gold standard per '{url}' aggiornato con successo!",
        )

    def insert_parsed_text(self, url, parsed_text) -> None:
        """
        Inserisce o aggiorna un parsed text nel database.

        args:
            url (str): URL della risorsa web a cui si riferisce il parsed text
            parsed_text (str): testo parsato da valutare
        """
        print(f"Inserimento/aggiornamento parsed text per URL: {url}")
        created_at = self._now()
        self._upsert(
            table="parsed_text",
            url=url,
            insert_sql="""
                INSERT INTO parsed_text (url, parsed_text, created_at)
                VALUES (?, ?, ?)
            """,
            insert_values=(url, parsed_text, created_at),
            update_sql="""
                UPDATE parsed_text
                SET parsed_text = ?, created_at = ?
                WHERE url = ?
            """,
            update_values=(parsed_text, created_at, url),
            insert_message="Parsed text per '{url}' inserito con successo!",
            update_message="Parsed text per '{url}' aggiornato con successo!",
        )

    def insert_evaluation(
        self,
        url,
        val_precision,
        val_recall,
        val_f1_score,
        val_lev_dist_norm,
        val_jac_sim_norm,
    ) -> None:
        """
        Inserisce o aggiorna una valutazione nel database.

        args:
            url (str): URL della risorsa web a cui si riferisce la valutazione
            val_precision (float): valore di precisione
            val_recall (float): valore di recall
            val_f1_score (float): valore di F1 score
            val_lev_dist_norm (float): valore di distanza di Levenshtein normalizzata
            val_jac_sim_norm (float): valore di similarità di Jaccard normalizzata
        """
        print(f"Inserimento/aggiornamento valutazione per URL: {url}")
        created_at = self._now()
        self._upsert(
            table="evaluation",
            url=url,
            insert_sql="""
                INSERT INTO evaluation (
                    url,
                    val_precision,
                    val_recall,
                    val_f1_score,
                    val_lev_dist_norm,
                    val_jac_sim_norm,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            insert_values=(
                url,
                val_precision,
                val_recall,
                val_f1_score,
                val_lev_dist_norm,
                val_jac_sim_norm,
                created_at,
            ),
            update_sql="""
                UPDATE evaluation
                SET val_precision = ?, val_recall = ?, val_f1_score = ?,
                    val_lev_dist_norm = ?, val_jac_sim_norm = ?, created_at = ?
                WHERE url = ?
            """,
            update_values=(
                val_precision,
                val_recall,
                val_f1_score,
                val_lev_dist_norm,
                val_jac_sim_norm,
                created_at,
                url,
            ),
            insert_message="Valutazione per '{url}' inserita con successo!",
            update_message="Valutazione per '{url}' aggiornata con successo!",
        )

    def insert_llm_evaluation(self, url, model, val_llm_score, val_llm_comment) -> None:
        """
        Inserisce o aggiorna una valutazione LLM nel database.

        args:
            url (str): URL della risorsa web a cui si riferisce la valutazione LLM
            model (str): nome del modello LLM utilizzato per la valutazione
            val_llm_score (float): valore di score LLM
            val_llm_comment (str): commento sulla valutazione LLM
        """
        print(f"Inserimento/aggiornamento valutazione LLM per URL: {url}")
        created_at = self._now()
        self._upsert(
            table="llm_eval",
            url=url,
            insert_sql="""
                INSERT INTO llm_eval (url, model_name, val_llm_score, val_llm_comment, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
            insert_values=(url, model, val_llm_score, val_llm_comment, created_at),
            update_sql="""
                UPDATE llm_eval
                SET model_name = ?, val_llm_score = ?, val_llm_comment = ?, created_at = ?
                WHERE url = ?
            """,
            update_values=(model, val_llm_score, val_llm_comment, created_at, url),
            insert_message="Valutazione LLM per '{url}' inserita con successo!",
            update_message="Valutazione LLM per '{url}' aggiornata con successo!",
        )

    def delete_web_resource(self, url) -> None:
        """
        Elimina una web resource dal database.

        args:
            url (str): URL della risorsa web da eliminare
        """
        print(f"Eliminazione web resource per URL: {url}")
        self._elimina(
            table="web_resources",
            url=url,
            message="Web resource '{url}' eliminata con successo!",
        )

    def delete_gold_standard(self, url) -> None:
        """
        Elimina un gold standard dal database.

        args:
            url (str): URL del gold standard da eliminare
        """
        self._elimina(
            table="gold_standard",
            url=url,
            message="Gold standard '{url}' eliminato con successo!",
        )

    def delete_parsed_text(self, url) -> None:
        """
        Elimina un parsed text dal database.

        args:
            url (str): URL del parsed text da eliminare
        """
        print(f"Eliminazione parsed text per URL: {url}")
        self._elimina(
            table="parsed_text",
            url=url,
            message="Parsed text '{url}' eliminato con successo!",
        )

    def delete_evaluation(self, url) -> None:
        """
        Elimina una valutazione dal database.

        args:
            url (str): URL della valutazione da eliminare
        """
        print(f"Eliminazione valutazione per URL: {url}")
        self._elimina(
            table="evaluation",
            url=url,
            message="Valutazione '{url}' eliminata con successo!",
        )

    def delete_llm_evaluation(self, url) -> None:
        """
        Elimina una valutazione LLM dal database.

        args:
            url (str): URL della valutazione LLM da eliminare
        """
        print(f"Eliminazione valutazione LLM per URL: {url}")
        self._elimina(
            table="llm_eval",
            url=url,
            message="Valutazione LLM '{url}' eliminata con successo!",
        )

    def delete_all(self) -> None:
        """
        Elimina tutte le risorse
        """
        with self._sessione() as cursor:
            cursor.execute("DELETE FROM web_resources")
            cursor.execute("DELETE FROM gold_standard")
            cursor.execute("DELETE FROM parsed_text")
            cursor.execute("DELETE FROM evaluation")
            cursor.execute("DELETE FROM llm_eval")
        print("Tutte le risorse sono state eliminate con successo!")


def main():
    #usa delete_all
    manipulator = DatabaseManipulator()
    inp = input("premi y per eliminare tutte le tabelle: ")
    if inp.lower() == "y":
        manipulator.delete_all()
    else:
        print("operazione annullata")

if __name__ == "__main__":
    main()