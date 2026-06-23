from dotenv import load_dotenv
import os
import mariadb  # type: ignore

load_dotenv()

URL_COLUMN = "VARCHAR(2048) CHARACTER SET ascii COLLATE ascii_general_ci"


def create_tables() -> None:
    """
    Crea le tabelle necessarie nel database MariaDB se non esistono già.
    """
    conn = mariadb.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
    )
    cursor = conn.cursor()

    # Creazione tabella Web Resources
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS web_resources (
            url {URL_COLUMN} PRIMARY KEY,
            domain VARCHAR(255),
            title VARCHAR(2048),
            html_text LONGTEXT,
            created_at DATETIME
        )
    """)
    print("Tabella web_resources creata con successo!")

    # Creazione tabella Gold Standard
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS gold_standard (
            url {URL_COLUMN} PRIMARY KEY,
            gold_text LONGTEXT,
            created_at DATETIME,
            CONSTRAINT fk_gold_standard 
                FOREIGN KEY (url) REFERENCES web_resources(url) 
                ON DELETE CASCADE
        )
    """)
    print("Tabella gold_standard creata con successo!")

    # Creazione tabella Parsed Text
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS parsed_text (
            url {URL_COLUMN} PRIMARY KEY,
            parsed_text LONGTEXT,
            created_at DATETIME,
            CONSTRAINT fk_parsed_text
                FOREIGN KEY (url) REFERENCES web_resources(url) 
                ON DELETE CASCADE
        )
    """)
    print("Tabella parsed_text creata con successo!")

    # Creazione tabella Evaluation
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS evaluation (
            url {URL_COLUMN} PRIMARY KEY,
            val_precision FLOAT,
            val_recall FLOAT,
            val_f1_score FLOAT,
            val_lev_dist_norm FLOAT,
            val_jac_sim_norm FLOAT,
            created_at DATETIME,
            CONSTRAINT fk_evaluation
                FOREIGN KEY (url) REFERENCES parsed_text(url) 
                ON DELETE CASCADE
        )
    """)
    print("Tabella evaluation creata con successo!")

    # Creazione tabella LLM Evaluation
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS llm_eval (
            url {URL_COLUMN} PRIMARY KEY,
            model_name VARCHAR(255),
            val_llm_score FLOAT,
            val_llm_comment LONGTEXT,
            created_at DATETIME,
            CONSTRAINT fk_llm_eval
                FOREIGN KEY (url) REFERENCES parsed_text(url) 
                ON DELETE CASCADE
        )
    """)
    print("Tabella llm_eval creata con successo!")

    conn.commit()
    cursor.close()
    conn.close()
    print("Tabelle create con successo!")


if __name__ == "__main__":
    create_tables()
