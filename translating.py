from datetime import datetime
import csv
import re
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import pandas as pd
from datetime import date
import spacy

nlp_es = spacy.load("es_core_news_sm")
def finde_lemma(spanisches_wort: str) -> str:
    """Gibt die Grundform (Infinitiv) eines spanischen Wortes zurück."""
    doc = nlp_es(spanisches_wort)
    for token in doc:
        return token.lemma_.lower()
    return spanisches_wort.lower()  # Fallback, falls kein Token erkannt wird



# === 1. MINI-INDEX vorbereiten ===
def build_test_index():
    docs = [
        Document(text="Spanisch: hola\nDeutsch: hallo", metadata={"kategorie_filter": "Grundlagen"}),
        Document(text="Spanisch: tiempo\nDeutsch: Zeit", metadata={"kategorie_filter": "Grundlagen"}),
        Document(text="Spanisch: café\nDeutsch: Kaffee", metadata={"kategorie_filter": "Alltag"}),
    ]
    index = VectorStoreIndex.from_documents(docs)
    return index

# === 2. MODELLE EINSTELLEN ===
Settings.llm = Ollama(
    model="phi3:3.8b-mini-4k-instruct-q4_K_M",  # klein und sparsam
    temperature=0.1,
    base_url="http://localhost:11434"
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def uebersetze_und_lerne(index, satz: str, csv_datei: str):
    """
    Übersetzt einen Satz und lernt neue Vokabeln automatisch hinzu.
    """
    print(f"\n--- Übersetze & Lerne: '{satz}' ---")

    # 1️⃣ Tokenisierung (vereinfachte Worterkennung)
    # Nur alphabetische Tokens (ignoriert Satzzeichen)
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÄÖÜß]+", satz)
    tokens = [t.lower() for t in tokens]

    # 2️⃣ Prüfen, ob Wörter im Index / CSV existieren
    vorhandene, neue = pruefe_vokabeln(csv_datei, tokens)

    # 3️⃣ Satzübersetzung (über LLM)
    uebersetzung, erklaerung = uebersetze_mit_llm(satz, neue)

    # 4️⃣ Neue Wörter automatisch zur CSV hinzufügen
    if neue:
        fuege_neue_vokabeln_hinzu(csv_datei, neue, satz)
        print("🆕 Neue Wörter erkannt und gespeichert:", ", ".join(neue))
    
    if vorhandene:
        print("✅ Bereits bekannte Wörter:", ", ".join(vorhandene))

    # 5️⃣ Ausgabe
    print("\n--- Übersetzung ---")
    print(uebersetzung)
    if erklaerung:
        print("\n--- Neue Vokabeln erklärt ---")
        print(erklaerung)
    else:
        print("\n✅ Alle Wörter waren bereits bekannt.")
    
    return uebersetzung


def pruefe_vokabeln(csv_datei: str, tokens: list[str]):
    """
    Prüft, welche Wörter bereits in der CSV-Vokabelliste enthalten sind.
    Lemmatisiert spanische Wörter, um Grundformen zu speichern.
    Gibt zwei Mengen zurück:
      - vorhandene: bekannte Wörter aus der CSV
      - neue: unbekannte Wörter, die ergänzt werden müssen
    """
    try:
        df = pd.read_csv(csv_datei, sep=';')
    except FileNotFoundError:
        print(f"⚠️  Datei '{csv_datei}' nicht gefunden – es wird angenommen, dass keine Vokabeln existieren.")
        df = pd.DataFrame(columns=["Spanisch", "Deutsch", "Kategorie", "Beispielsatz", "last_repetition"])

    vorhandene = []
    neue = []

    for wort in tokens:
        lemma = finde_lemma(wort.lower())

        if lemma in df['Spanisch'].astype(str).str.lower().values:
            vorhandene.append(lemma)
        else:
            neue.append(lemma)

    return vorhandene, neue


def uebersetze_mit_llm(satz, neue_vokabeln):
    """
    Übersetzt den Satz und gibt Erklärungen zu unbekannten Wörtern.
    """
    prompt = (
        "Du bist ein Spanisch-Deutsch-Übersetzer und Sprachlehrer.\n"
        "Übersetze den folgenden Satz vollständig und natürlich.\n"
        "Falls einige Wörter nicht bekannt sind oder schwierig sein könnten, "
        "gib danach eine kurze Erklärung auf Deutsch (Bedeutung und ggf. Beispiel).\n\n"
        f"Satz: {satz}\n\n"
    )
    if neue_vokabeln:
        prompt += f"Diese Wörter sind neu: {', '.join(neue_vokabeln)}\n"

    # Direkte LLM-Abfrage ohne Index
    try:
        response = Settings.llm.complete(prompt)
        text = response.text
    except Exception as e:
        text = f"Fehler bei der Übersetzung: {e}"

    # Optional: Split in Übersetzung und Erklärung
    parts = text.split("\n\n", 1)
    translation = parts[0].strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""

    return translation, explanation

def fuege_neue_vokabeln_hinzu(csv_datei, neue_worter, original_satz):
    """
    Fügt neue Wörter in die CSV ein mit LLM-generierten Übersetzungen und Kategorien.
    """
    heute = datetime.now().strftime("%Y-%m-%d")
    
    # LLM-Prompt für jede neue Vokabel
    for wort in neue_worter:
        prompt = (
            f"Du bist ein Spanisch-Lehrer. Analysiere das spanische Wort '{wort}' im Kontext des Satzes:\n"
            f"'{original_satz}'\n\n"
            f"Gib folgende Informationen zurück (genau in diesem Format):\n"
            f"DEUTSCH: [NUR EIN deutsches Wort als Übersetzung, keine Erklärung]\n"
            f"KATEGORIE: [passende Kategorie wie 'Alltag', 'Verben', 'Adjektive', 'Grundlagen', etc.]\n\n"
            f"Wichtig: Bei DEUTSCH nur ein einzelnes Wort angeben, keine Sätze oder Erklärungen!\n"
            f"Antworte NUR mit diesen zwei Zeilen, keine zusätzlichen Erklärungen."
        )
        
        try:
            response = Settings.llm.complete(prompt)
            text = response.text.strip()
            
            # Parse die Antwort
            deutsch = ""
            kategorie = "Unbekannt"
            
            for line in text.split("\n"):
                if line.startswith("DEUTSCH:"):
                    deutsch = line.replace("DEUTSCH:", "").strip()
                    # Nur das erste Wort nehmen, falls mehrere zurückgegeben wurden
                    deutsch = deutsch.split()[0] if deutsch else ""
                elif line.startswith("KATEGORIE:"):
                    kategorie = line.replace("KATEGORIE:", "").strip()
            
            # Fallback, falls Parsing fehlschlägt
            if not deutsch:
                deutsch = f"[{wort}]"
                
        except Exception as e:
            print(f"⚠️  Fehler beim Abrufen der Übersetzung für '{wort}': {e}")
            deutsch = ""
            kategorie = "Unbekannt"
        
        # Zur CSV hinzufügen - Reihenfolge muss mit CSV-Spalten übereinstimmen
        neue_zeile = {
            "Spanisch": wort,
            "Deutsch": deutsch,
            "Beispielsatz": original_satz,
            "last_repetition": heute,
            "Kategorie": kategorie
            
        }
        
        # CSV erweitern
        try:
            df = pd.read_csv(csv_datei, sep=';')
            
            # Entferne leere Zeilen (alle Spalten sind leer oder NaN)
            df = df.dropna(how='all')
            
            # Prüfe, welche Spalte für das Datum existiert (mit oder ohne Unterstrich)
            if 'lastrepetition' in df.columns:
                # Umbenennen: lastrepetition -> last_repetition
                df = df.rename(columns={'lastrepetition': 'last_repetition'})
            
            # Stelle sicher, dass die Spalten in der richtigen Reihenfolge sind
            spalten_reihenfolge = ["Spanisch", "Deutsch", "Beispielsatz", "last_repetition", "Kategorie"]
            
            # Prüfe, ob alle benötigten Spalten vorhanden sind
            fehlende_spalten = [col for col in spalten_reihenfolge if col not in df.columns]
            if fehlende_spalten:
                print(f"⚠️  Warnung: Fehlende Spalten in CSV: {fehlende_spalten}")
                # Füge fehlende Spalten mit leeren Werten hinzu
                for col in fehlende_spalten:
                    df[col] = ""
            
            # Falls die CSV andere Spalten hat, ordne sie neu an
            df = df[spalten_reihenfolge]
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Spanisch", "Deutsch", "Beispielsatz", "last_repetition", "Kategorie"])
        
        # Neue Zeile hinzufügen mit expliziter Spaltenreihenfolge
        neue_zeile_df = pd.DataFrame([neue_zeile], columns=["Spanisch", "Deutsch", "Beispielsatz", "last_repetition", "Kategorie"])
        df = pd.concat([df, neue_zeile_df], ignore_index=True)
        
        # Nochmal leere Zeilen entfernen vor dem Speichern
        df = df.dropna(how='all')
        
        # Sicherstellen, dass beim Speichern die Spaltenreihenfolge erhalten bleibt
        df = df[["Spanisch", "Deutsch", "Beispielsatz", "last_repetition", "Kategorie"]]
        df.to_csv(csv_datei, sep=';', index=False)
        
        print(f"   ✓ {wort} → {deutsch} ({kategorie})")
    
    print(f"🆕 {len(neue_worter)} neue Vokabel(n) mit Übersetzung hinzugefügt.")


if __name__ == "__main__":

    index = build_test_index()
    uebersetze_und_lerne(index,satz="Ellos estudian ciencias.",csv_datei="../vokabeln.csv")