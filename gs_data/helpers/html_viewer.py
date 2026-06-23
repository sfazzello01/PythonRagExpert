
#script per visualizzare l'HTML estratto da un file JSON in un browser,
# con gestione del dominio originale per risolvere i link relativi

import json
import os
import webbrowser
import html
import re

# 1. Chiede interattivamente il percorso del file JSON
json_file_path = input(
    "Inserisci il percorso (assoluto o relativo) del file JSON: "
).strip()
json_file_path = json_file_path.strip("'\"")

# Nome del file HTML temporaneo
temp_html_path = "temp_output.html"

try:
    # 2. Carica il file JSON
    with open(json_file_path, "r", encoding="utf-8") as file:
        dati = json.load(file)

    # 3. Estrai l'HTML usando la struttura annidata
    gold_standard = dati.get("gold_standard")
    if gold_standard and len(gold_standard) > 0:
        html_content = gold_standard[0].get("html_text")
    else:
        html_content = None

    if html_content:
        # Pulizia iniziale dei caratteri speciali
        html_content = html.unescape(html_content).strip()

        # --- STRATEGIA: RECUPERO DEL DOMINIO ORIGINALE ---
        # Estrae "www.borsaitaliana.it" da "/percorso/.../www.borsaitaliana.it.json"
        nome_file = os.path.basename(json_file_path)
        dominio = nome_file.replace(".json", "")

        # Se il nome del file assomiglia a un dominio usa quello, altrimenti usa borsaitaliana come fallback
        if "." in dominio and not dominio.startswith("temp"):
            base_url = f"https://{dominio}/"
        else:
            base_url = "https://www.borsaitaliana.it/"

        print(f"🔮 Dominio sorgente rilevato: {base_url}")

        # --- INIEZIONE DEL TAG <base> ---
        # Cerchiamo il tag <head> o <html> usando le espressioni regolari per gestire gli attributi
        if re.search(r"<head[^>]*>", html_content, re.IGNORECASE):
            html_content = re.sub(
                r"(<head[^>]*>)",
                f"\\1\n    <base href='{base_url}'>",
                html_content,
                1,
                flags=re.IGNORECASE,
            )
            print("🚀 Tag <base> iniettato dentro <head>.")
        elif re.search(r"<html[^>]*>", html_content, re.IGNORECASE):
            html_content = re.sub(
                r"(<html[^>]*>)",
                f"\\1\n    <base href='{base_url}'>",
                html_content,
                1,
                flags=re.IGNORECASE,
            )
            print("🚀 Tag <base> iniettato dopo <html>.")
        else:
            # Se mancano sia html che head, lo inseriamo all'inizio del file
            html_content = f"<base href='{base_url}'>\n" + html_content
            print("🚀 Tag <base> inserito in testa al file.")

        # 4. Salva il contenuto HTML modificato nel file locale
        with open(temp_html_path, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)

        # 5. Ottieni il percorso assoluto e aprilo nel browser
        path_assoluto = os.path.abspath(temp_html_path)
        webbrowser.open(f"file://{path_assoluto}")

        print(
            f"\n[Successo] Il file HTML è stato aperto"
        )
    else:
        print("\n[Errore] La chiave 'html_text' non è stata trovata o è vuota.")

except FileNotFoundError:
    print(f"\n[Errore] Non ho trovato nessun file al percorso: '{json_file_path}'.")
except json.JSONDecodeError:
    print("\n[Errore] Il file selezionato non è un JSON valido.")
