# Seed per popolare il database con i dati dei gold standard.

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import mariadb  # type: ignore

from ..utilities import iter_gs_entries

load_dotenv()


def seed_mariadb() -> None:
    """
    Popola il database MariaDB con i dati dei gold standard.
    """
    config = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
    }

    connection = None
    cursor = None

    try:
        # Connessione usando la config estratta dal .env
        connection = mariadb.connect(**config)
        cursor = connection.cursor()

        created_at = datetime.now(ZoneInfo("Europe/Rome"))

        # Carica tutte le entry dei gold standard.
        for _, entry in iter_gs_entries():
            cursor.execute(
                """
                INSERT INTO web_resources (url, domain, title, html_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    domain = VALUES(domain),
                    title = VALUES(title),
                    html_text = VALUES(html_text),
                    created_at = VALUES(created_at)
                """,
                (
                    entry["url"],
                    entry["domain"],
                    entry["title"],
                    entry["html_text"],
                    created_at,
                ),
            )

            cursor.execute(
                """
                INSERT INTO gold_standard (url, gold_text, created_at)
                VALUES (?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    gold_text = VALUES(gold_text),
                    created_at = VALUES(created_at)
                """,
                (
                    entry["url"],
                    entry["gold_text"],
                    created_at,
                ),
            )
        connection.commit()
        print("Dati dei gold standard inseriti con successo!")

        print("Connessione sicura stabilita!")

    except Exception as e:
        if connection is not None:
            connection.rollback()
        print(f"Errore: {e}")

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    seed_mariadb()
