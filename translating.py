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
    """
    Gibt die Grundform (Infinitiv oder Basisform) eines spanischen Wortes zurück.
    Spezielle Behandlung für Pronomen, Akzentwörter und häufige Formen.
    """
    # Akzent-sensitives Mapping (um Fehlinterpretationen zu vermeiden)
    sonderformen = {
        "él": "él",   # er
        "el": "el",   # der
        "sí": "sí",   # ja
        "si": "si",   # wenn
        "tú": "tú",   # du
        "tu": "tu",   # dein
        "mí": "mí",   # mir / mich
        "mi": "mi",   # mein
        "sé": "sé",   # ich weiß / sei
        "se": "se",   # sich
        "dé": "dé",   # gebe (Subjunktiv)
        "de": "de"    # von
    }

    pronomen_mapping = {
        "yo": "yo",
        "tú": "tú",
        "usted": "usted",
        "él": "él",
        "ella": "ella",
        "ello": "ello",
        "nosotros": "nosotros",
        "nosotras": "nosotros",
        "vosotros": "vosotros",
        "vosotras": "vosotros",
        "ellos": "ellos",
        "ellas": "ellos",
        "ustedes": "ustedes",
        "me": "me",
        "te": "te",
        "se": "se",
        "lo": "lo",
        "la": "la",
        "los": "los",
        "las": "los",
        "le": "le",
        "les": "le"
    }

    # Wichtig: strip() verwenden, aber NICHT lower() bei Akzentwörtern
    wort_stripped = spanisches_wort.strip()
    wort_klein = wort_stripped.lower()

    # 1️⃣ Zuerst mit Original-Schreibweise (inkl. Akzente) prüfen
    if wort_stripped in sonderformen:
        return sonderformen[wort_stripped]
    
    if wort_stripped in pronomen_mapping:
        return pronomen_mapping[wort_stripped]

    # 2️⃣ Dann mit Kleinbuchstaben prüfen
    if wort_klein in sonderformen:
        return sonderformen[wort_klein]

    if wort_klein in pronomen_mapping:
        return pronomen_mapping[wort_klein]

    # 3️⃣ Lemmatisieren mit spaCy
    doc = nlp_es(wort_klein)
    for token in doc:
        # WICHTIG: Substantive NICHT lemmatisieren!
        # química (Chemie) != químico (Chemiker/chemisch)
        if token.pos_ == "NOUN":
            return wort_klein
        
        # Bei Pronomen das Original-Wort zurückgeben
        if token.pos_ == "PRON":
            return wort_klein
            
        lemma = token.lemma_.lower()

        # Korrigiere fehlerhafte spaCy-Lemmata für bekannte Sonderfälle
        if wort_klein in sonderformen and lemma != sonderformen[wort_klein]:
            return sonderformen[wort_klein]

        return lemma

    # 4️⃣ Fallback
    return wort_klein



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

    # 1️⃣ Tokenisierung mit spaCy (um Wortarten zu erkennen)
    doc = nlp_es(satz)
    
    # Filtere nur Satzzeichen und Leerzeichen heraus
    IGNORIERE_WORTARTEN = {"PUNCT", "SPACE"}
    
    # Speichere Tokens mit ihren Wortarten basierend auf Satz-Kontext
    token_info = {}  # {lemma: wortart}
    tokens = []
    for token in doc:
        # Ignoriere nur Satzzeichen und Leerzeichen
        if token.pos_ not in IGNORIERE_WORTARTEN:
            wort_lower = token.text.lower()
            
            # WICHTIG: Nutze Satz-Kontext für korrekte Lemmatisierung
            # Substantive NICHT lemmatisieren (química bleibt química)
            if token.pos_ == "NOUN":
                lemma = wort_lower
            elif token.pos_ == "PRON":
                lemma = wort_lower
            else:
                # Nur Verben, Adjektive etc. lemmatisieren
                lemma = token.lemma_.lower()
            
            tokens.append(lemma)
            token_info[lemma] = token.pos_  # Speichere Wortart mit Lemma als Schlüssel

    # 2️⃣ Prüfen, ob Wörter im Index / CSV existieren (mit Original-Satz für Kontext)
    vorhandene, neue = pruefe_vokabeln(csv_datei, tokens, satz)

    # 2.5️⃣ Erstelle korrigiertes token_info mit Original-Formen
    token_info_korrigiert = {}
    neue_original_liste = []
    lemma_to_original_map = {}  # Mapping: Lemma → Original-Form
    
    if neue:
        doc_satz = nlp_es(satz)
        for wort in neue:
            for token in doc_satz:
                if token.lemma_.lower() == wort.lower() or token.text.lower() == wort.lower():
                    original_form = token.text.lower()
                    # Speichere Mapping
                    lemma_to_original_map[wort] = original_form
                    token_info_korrigiert[wort] = token.pos_  # Lemma als Key, aber mit korrekter POS aus Satz!
                    neue_original_liste.append(original_form)
                    break
            # Falls nicht gefunden, verwende das lemmatisierte Wort
            if wort not in lemma_to_original_map:
                lemma_to_original_map[wort] = wort
                token_info_korrigiert[wort] = token_info.get(wort, "UNKNOWN")
                neue_original_liste.append(wort)

    # 3️⃣ Satzübersetzung (über LLM) - mit Original-Formen!
    uebersetzung, erklaerung = uebersetze_mit_llm(satz, neue_original_liste if neue else [])

    # 4️⃣ Neue Wörter automatisch zur CSV hinzufügen (mit korrigiertem token_info)
    if neue:
        fuege_neue_vokabeln_hinzu(csv_datei, neue, satz, token_info_korrigiert, lemma_to_original_map)
        print("🆕 Neue Wörter erkannt und gespeichert:", ", ".join(neue_original_liste))
    
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


def pruefe_vokabeln(csv_datei: str, tokens: list[str], original_satz: str = None):
    """
    Prüft, welche Wörter bereits in der CSV-Vokabelliste enthalten sind.
    Lemmatisiert spanische Wörter basierend auf dem Kontext im Original-Satz.
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
    
    # Wenn Original-Satz vorhanden, analysiere ihn für bessere Lemmatisierung
    satz_tokens = {}
    if original_satz:
        doc_satz = nlp_es(original_satz)
        for token in doc_satz:
            wort_lower = token.text.lower()
            # Speichere: Wort → (POS, Lemma)
            # Für Substantive: behalte Original-Form
            if token.pos_ == "NOUN":
                satz_tokens[wort_lower] = (token.pos_, wort_lower)
            elif token.pos_ == "PRON":
                satz_tokens[wort_lower] = (token.pos_, wort_lower)
            else:
                satz_tokens[wort_lower] = (token.pos_, token.lemma_.lower())

    for wort in tokens:
        # Nutze Kontext-basierte Lemmatisierung, falls verfügbar
        if wort.lower() in satz_tokens:
            _, lemma = satz_tokens[wort.lower()]
        else:
            # Fallback: Standard-Lemmatisierung
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

def fuege_neue_vokabeln_hinzu(csv_datei, neue_worter, original_satz, token_info=None, lemma_to_original=None):
    """
    Fügt neue Wörter in die CSV ein mit LLM-generierten Übersetzungen und Kategorien.
    token_info: Dictionary mit {lemma: wortart_im_satz_kontext}
    lemma_to_original: Dictionary mit {lemma: original_form_im_satz}
    """
    heute = datetime.now().strftime("%Y-%m-%d")
    
    # Wortart-Mapping für bessere Prompts
    WORTART_DEUTSCH = {
        "DET": "Artikel",
        "NOUN": "Substantiv",
        "VERB": "Verb",
        "ADJ": "Adjektiv",
        "ADV": "Adverb",
        "PRON": "Pronomen",
        "ADP": "Präposition",
        "CCONJ": "Konjunktion",
        "SCONJ": "Subjunktion"
    }
    
    # LLM-Prompt für jede neue Vokabel
    for wort in neue_worter:
        # WICHTIG: Verwende die Original-Form aus dem Mapping
        original_wort = lemma_to_original.get(wort, wort) if lemma_to_original else wort
        wortart_im_satz = token_info.get(wort, "UNKNOWN") if token_info else "UNKNOWN"
        
        print(f"   [DEBUG] Lemma '{wort}' → Original: '{original_wort}' (POS: {wortart_im_satz})")
        
        # Wortart herausfinden
        wortart_deutsch = WORTART_DEUTSCH.get(wortart_im_satz, "Wort")
        
        prompt = (
            f"Du bist ein Spanisch-Lehrer. Analysiere das spanische Wort '{original_wort}' im Kontext des Satzes:\n"
            f"'{original_satz}'\n\n"
            f"WICHTIG: '{original_wort}' ist ein {wortart_deutsch} ({wortart_im_satz})!\n\n"
            f"Gib folgende Informationen zurück (genau in diesem Format):\n"
            f"DEUTSCH: [NUR EIN deutsches Wort als Übersetzung, keine Erklärung]\n"
            f"KATEGORIE: [passende Kategorie wie 'Alltag', 'Verben', 'Adjektive', 'Artikel', 'Präpositionen', etc.]\n\n"
            f"Wichtig: Bei DEUTSCH nur ein einzelnes Wort angeben, keine Sätze oder Erklärungen!\n"
            f"Beispiele:\n"
            f"- 'el' (Artikel) → DEUTSCH: der\n"
            f"- 'la' (Artikel) → DEUTSCH: die\n"
            f"- 'de' (Präposition) → DEUTSCH: von\n"
            f"- 'en' (Präposition) → DEUTSCH: in\n\n"
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
                deutsch = f"[{original_wort}]"
            
            # KORREKTUR: Wenn deutsches Wort großgeschrieben ist → es ist ein Substantiv!
            # Überschreibe falsche spaCy-Klassifikation
            if deutsch and len(deutsch) > 0 and deutsch[0].isupper():
                # Deutsche Substantive sind IMMER großgeschrieben
                print(f"   [KORREKTUR] '{deutsch}' ist großgeschrieben → '{original_wort}' ist ein Substantiv!")
                kategorie = "Substantive"  # Korrigiere Kategorie
                
        except Exception as e:
            print(f"⚠️  Fehler beim Abrufen der Übersetzung für '{original_wort}': {e}")
            deutsch = ""
            kategorie = "Unbekannt"
        
        # Zur CSV hinzufügen - verwende ORIGINAL_WORT statt wort
        neue_zeile = {
            "Spanisch": original_wort,  # Verwende Original-Form aus dem Satz!
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
        
        print(f"   ✓ {original_wort} → {deutsch} ({kategorie})")
    
    print(f"🆕 {len(neue_worter)} neue Vokabel(n) mit Übersetzung hinzugefügt.")


if __name__ == "__main__":

    index = build_test_index()
    uebersetze_und_lerne(index,satz="él estudia química.",csv_datei="../vokabeln.csv")