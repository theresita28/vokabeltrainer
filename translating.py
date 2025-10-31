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

try:
    import simplemma
    USE_SIMPLEMMA = True
    print("✓ simplemma verfügbar → werde simplemma für Lemmatization verwenden.")
except Exception:
    USE_SIMPLEMMA = False
    print("⚠ simplemma nicht verfügbar → benutze spaCy-Fallback mit Korrekturen.")

nlp_es = spacy.load("es_core_news_sm")



# ⚠️ KORREKTUR-MAP für fehlerhafte spaCy-Lemmatisierungen
SPACY_LEMMA_KORREKTUREN = {
    "comar": "comer",      # spaCy Bug: come/comen werden fälschlicherweise zu "comar"
    "aprendo": "aprender", # spaCy Bug: aprendo bleibt aprendo statt aprender
    # Hier können weitere fehlerhafte Lemmas hinzugefügt werden
}

def finde_lemma(spanisches_wort: str) -> str:
    """
    Robust: Verwende pattern.es, falls verfügbar; sonst spaCy + Korrekturen + Heuristik.
    Liefert: lemma / Grundform (bei Verben Infinitiv), behandelt Akzente + Pronomen.
    """
    if not spanisches_wort:
        return spanisches_wort

    wort_orig = spanisches_wort.strip()
    wort = wort_orig.lower()

    # 1) Sonderfälle / Akzent-sensitive Mapping (wie du schon hattest)
    sonderformen = {
        "él": "él", "el": "el", "sí": "sí", "si": "si", "tú": "tú", "tu": "tu",
        "mí": "mí", "mi": "mi", "sé": "sé", "se": "se", "dé": "dé", "de": "de"
    }
    pronomen_mapping = {
        "yo":"yo","tú":"tú","usted":"usted","él":"él","ella":"ella","ello":"ello",
        "nosotros":"nosotros","nosotras":"nosotros","vosotros":"vosotros","vosotras":"vosotros",
        "ellos":"ellos","ellas":"ellos","ustedes":"ustedes","me":"me","te":"te","se":"se",
        "lo":"lo","la":"la","los":"los","las":"las","le":"le","les":"les"
    }

    # Prüfe Original (mit Akzent) zuerst
    if wort_orig in sonderformen:
        return sonderformen[wort_orig]

    if wort_orig in pronomen_mapping:
        return pronomen_mapping[wort_orig]

    # Prüfe Kleinschreibung
    if wort in sonderformen:
        return sonderformen[wort]
    if wort in pronomen_mapping:
        return pronomen_mapping[wort]

    # 2) Wenn simplemma verfügbar → verwenden (sehr zuverlässig für Verben)
    simplemma_result = None
    if USE_SIMPLEMMA:
        try:
            simplemma_result = simplemma.lemmatize(wort, lang='es')
            # Wenn simplemma eine Änderung vornimmt, vertrauen wir dem Ergebnis
            if simplemma_result and simplemma_result != wort:
                # Prüfe ob es ein Verb-Infinitiv ist (endet auf -ar, -er, -ir)
                if simplemma_result.endswith(('ar', 'er', 'ir')):
                    return simplemma_result.lower()  # Definitiv ein Verb!
                # Auch bei anderen Änderungen vertrauen wir simplemma
                return simplemma_result.lower()
            # Wenn simplemma KEINE Änderung macht, weitermachen mit spaCy + Heuristik
        except Exception:
            # falls simplemma aus irgendeinem Grund versagt, fall through
            pass

    # 3) spaCy-Fallback: benutze POS & Lemma, aber korrigiere bekannte spaCy-Fehler
    lemma = None
    try:
        doc = nlp_es(wort)
        for token in doc:
            # Bei echten Pronomen: Originalform beizubehalten
            if token.pos_ == "PRON":
                lemma = wort
            else:
                # Vertraue spaCy's Lemma (wird ggf. später korrigiert)
                lemma = token.lemma_.lower()
            break
    except Exception:
        lemma = wort

    # 4) Korrektur-Map von spaCy-Bugs anwenden (deine Map)
    if lemma in SPACY_LEMMA_KORREKTUREN:
        return SPACY_LEMMA_KORREKTUREN[lemma]

    # 5) Wenn lemma == wort (weder simplemma noch spaCy half), heuristische Fallbacks
    if lemma == wort or not lemma:
        # Einfache heuristiken für gängige spanische Konjugationen:
        if wort.endswith("o") and len(wort) > 2:
            # estudio → estudiar, hablo → hablar
            return wort[:-1] + "ar"
        if wort.endswith(("as","a","amos","an")):
            return (wort[:-1] + "ar") if wort.endswith(("as","a")) else (wort[:-4] + "ar") if wort.endswith("amos") else wort[:-2] + "ar"
        if wort.endswith(("es","e","emos","en")):
            return (wort[:-2] + "er") if wort.endswith("es") else (wort[:-1] + "er")
        if wort.endswith(("imos","en","ieron","ió","ió")):
            return wort[:-2] + "ir"
        # sonst zurückgeben
        return wort

    return lemma



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
    base_url="http://localhost:11434",
    request_timeout=30.0  # 30 Sekunden Timeout
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
            # Substantive → Einzahl (mit simplemma falls verfügbar)
            if token.pos_ == "NOUN":
                if USE_SIMPLEMMA:
                    lemma_simplemma = simplemma.lemmatize(wort_lower, lang='es')
                    # Fallback auf spaCy, wenn simplemma keine Änderung vornimmt
                    if lemma_simplemma == wort_lower and token.lemma_.lower() != wort_lower:
                        lemma = token.lemma_.lower()
                    elif lemma_simplemma == wort_lower:
                        # Weder simplemma noch spaCy half
                        # Nur Heuristik anwenden wenn es WIRKLICH ein falsch klassifiziertes Verb sein könnte
                        # Hinweis: hermano, mano, año etc. sind echte Substantive!
                        # Nur bei bekannten Verb-Mustern: estudio, hablo, trabajo etc.
                        bekannte_verben_auf_o = ["estudio", "trabajo", "hablo", "como", "vivo", "escribo", "leo"]
                        if wort_lower in bekannte_verben_auf_o:
                            lemma = wort_lower[:-1] + "ar"  # estudio → estudiar
                        else:
                            lemma = wort_lower  # hermano bleibt hermano
                    else:
                        lemma = lemma_simplemma
                else:
                    lemma = wort_lower  # Fallback: behalte Original
            elif token.pos_ == "PRON":
                lemma = wort_lower  # Pronomen nicht ändern
            else:
                # Verben, Adjektive etc. → Grundform (mit simplemma falls verfügbar)
                if USE_SIMPLEMMA:
                    lemma_simplemma = simplemma.lemmatize(wort_lower, lang='es')
                    # Fallback auf spaCy, wenn simplemma keine Änderung vornimmt
                    if lemma_simplemma == wort_lower and token.lemma_.lower() != wort_lower:
                        # simplemma kennt das Wort nicht → nutze spaCy (mit Korrekturen)
                        lemma = token.lemma_.lower()
                        if lemma in SPACY_LEMMA_KORREKTUREN:
                            lemma = SPACY_LEMMA_KORREKTUREN[lemma]
                    else:
                        lemma = lemma_simplemma
                else:
                    # Fallback: spaCy + Korrekturen
                    lemma = token.lemma_.lower()
                    if lemma in SPACY_LEMMA_KORREKTUREN:
                        lemma = SPACY_LEMMA_KORREKTUREN[lemma]
            
            tokens.append(lemma)
            token_info[lemma] = token.pos_  # Speichere Wortart mit Lemma als Schlüssel

    # 2️⃣ Prüfen, ob Wörter im Index / CSV existieren (mit Original-Satz für Kontext)
    vorhandene, neue = pruefe_vokabeln(csv_datei, tokens, satz)

    # 2.5️⃣ Erstelle korrigiertes token_info mit Original-Formen (für LLM-Erklärung)
    token_info_korrigiert = {}
    neue_original_liste = []  # Für LLM-Erklärung (Satzform)
    lemma_to_satzform_map = {}  # Mapping: Lemma → Konjugierte Form im Satz (für Erklärung)
    
    if neue:
        doc_satz = nlp_es(satz)
        for wort in neue:
            for token in doc_satz:
                if token.lemma_.lower() == wort.lower() or token.text.lower() == wort.lower():
                    satzform = token.text.lower()  # Konjugierte Form (für Erklärung)
                    # Speichere Mapping (für LLM-Erklärung)
                    lemma_to_satzform_map[wort] = satzform
                    token_info_korrigiert[wort] = token.pos_  # Lemma als Key, aber mit korrekter POS aus Satz!
                    neue_original_liste.append(satzform)
                    break
            # Falls nicht gefunden, verwende das lemmatisierte Wort
            if wort not in lemma_to_satzform_map:
                lemma_to_satzform_map[wort] = wort
                token_info_korrigiert[wort] = token_info.get(wort, "UNKNOWN")
                neue_original_liste.append(wort)

    # 3️⃣ Neue Wörter automatisch zur CSV hinzufügen (mit korrigiertem token_info) 
    # UND deutsche Übersetzungen sammeln
    vokabel_uebersetzungen = {}  # {spanisch_satzform: deutsch}
    if neue:
        vokabel_uebersetzungen = fuege_neue_vokabeln_hinzu(csv_datei, neue, satz, token_info_korrigiert, lemma_to_satzform_map)
        print("🆕 Neue Wörter erkannt und gespeichert:", ", ".join(neue))  # Zeige Lemmas/Grundformen
    
    # 3.5️⃣ Hole auch Übersetzungen für bereits bekannte Wörter aus CSV
    if vorhandene:
        try:
            df = pd.read_csv(csv_datei, sep=';')
            doc_satz = nlp_es(satz)
            for wort in vorhandene:
                # Finde deutsche Übersetzung in CSV
                matching = df[df['Spanisch'].astype(str).str.lower() == wort.lower()]
                if not matching.empty:
                    deutsch = matching.iloc[0]['Deutsch']
                    # Finde Satzform für dieses Wort
                    for token in doc_satz:
                        if token.lemma_.lower() == wort.lower() or token.text.lower() == wort.lower():
                            vokabel_uebersetzungen[token.text.lower()] = deutsch
                            break
        except Exception as e:
            print(f"⚠️  Fehler beim Laden bekannter Übersetzungen: {e}")
    
    # 4️⃣ Satzübersetzung (über LLM) - mit Original-Formen UND deutschen Übersetzungen!
    uebersetzung, erklaerung = uebersetze_mit_llm(satz, neue_original_liste if neue else [], vokabel_uebersetzungen)
    
    if vorhandene:
        print("✅ Bereits bekannte Wörter:", ", ".join(vorhandene))
    
    # Abschließende Zusammenfassung
    if not neue and not vorhandene:
        print("✅ Alle Wörter waren bereits bekannt.")

    # 5️⃣ Ausgabe
    print("\n--- Übersetzung ---")
    print(uebersetzung)
    if erklaerung:
        print("\n--- Neue Vokabeln erklärt ---")
        print(erklaerung)
    
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
            # Für Substantive und Pronomen: verwende simplemma (falls verfügbar)
            if token.pos_ == "NOUN":
                if USE_SIMPLEMMA:
                    lemma_simplemma = simplemma.lemmatize(wort_lower, lang='es')
                    # Fallback auf spaCy, wenn simplemma keine Änderung vornimmt
                    if lemma_simplemma == wort_lower and token.lemma_.lower() != wort_lower:
                        lemma = token.lemma_.lower()
                    else:
                        lemma = lemma_simplemma
                else:
                    lemma = wort_lower
                satz_tokens[wort_lower] = (token.pos_, lemma)
            elif token.pos_ == "PRON":
                satz_tokens[wort_lower] = (token.pos_, wort_lower)
            else:
                # Verben, Adjektive etc.: verwende simplemma (falls verfügbar)
                if USE_SIMPLEMMA:
                    lemma_simplemma = simplemma.lemmatize(wort_lower, lang='es')
                    # Fallback auf spaCy, wenn simplemma keine Änderung vornimmt
                    if lemma_simplemma == wort_lower and token.lemma_.lower() != wort_lower:
                        lemma = token.lemma_.lower()
                    else:
                        lemma = lemma_simplemma
                else:
                    lemma = token.lemma_.lower()
                satz_tokens[wort_lower] = (token.pos_, lemma)

    for wort in tokens:
        # "wort" ist bereits das korrekte Lemma aus der Tokenisierung!
        # Keine erneute Lemmatisierung nötig
        lemma = wort

        if lemma in df['Spanisch'].astype(str).str.lower().values:
            vorhandene.append(lemma)
        else:
            neue.append(lemma)

    return vorhandene, neue


def uebersetze_mit_llm(satz, neue_vokabeln, vokabel_uebersetzungen=None):
    """
    Übersetzt den Satz und gibt Erklärungen zu unbekannten Wörtern.
    vokabel_uebersetzungen: Dictionary {spanisch_satzform: deutsch} für korrekte Erklärungen
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
    
    # Wenn deutsche Übersetzungen verfügbar sind, füge sie dem Prompt hinzu
    if vokabel_uebersetzungen:
        prompt += "\nBekannte Übersetzungen (VERWENDE DIESE!):\n"
        for span, deu in vokabel_uebersetzungen.items():
            prompt += f"- {span} = {deu}\n"
        prompt += "\nWICHTIG: Verwende genau diese Übersetzungen in deiner Satzübersetzung und Erklärung!\n"

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

def fuege_neue_vokabeln_hinzu(csv_datei, neue_worter, original_satz, token_info=None, lemma_to_satzform=None):
    """
    Fügt neue Wörter in die CSV ein mit LLM-generierten Übersetzungen und Kategorien.
    token_info: Dictionary mit {lemma: wortart_im_satz_kontext}
    lemma_to_satzform: Dictionary mit {lemma: konjugierte_form_im_satz} für bessere LLM-Prompts
    
    Gibt zurück: Dictionary {satzform: deutsch} für die LLM-Satzübersetzung
    """
    heute = datetime.now().strftime("%Y-%m-%d")
    uebersetzungen = {}  # Sammle deutsche Übersetzungen für LLM
    
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
        # wort = Lemma/Grundform (z.B. "comer")
        # Für den LLM-Prompt: Verwende konjugierte Form bei Erklärung
        satzform = lemma_to_satzform.get(wort, wort) if lemma_to_satzform else wort
        wortart_im_satz = token_info.get(wort, "UNKNOWN") if token_info else "UNKNOWN"
        
        print(f"   [DEBUG] Lemma '{wort}' → Satzform: '{satzform}' (POS: {wortart_im_satz})")
        
        # Wortart herausfinden
        wortart_deutsch = WORTART_DEUTSCH.get(wortart_im_satz, "Wort")
        
        # Bei Verben UND Substantiven: Verwende das LEMMA für den Prompt (Grundform/Infinitiv/Einzahl)
        # Nur bei Pronomen/Artikeln: verwende die Satzform
        if wortart_im_satz in ["VERB", "NOUN", "ADJ"]:
            wort_fuer_prompt = wort  # Lemma/Grundform
        else:
            wort_fuer_prompt = satzform  # z.B. bei Pronomen: "él", "ella"
        
        # Spezielle Anweisungen für Verben (Infinitiv-Form verlangen)
        verb_hinweis = ""
        if wortart_im_satz == "VERB":
            verb_hinweis = "\nWICHTIG: Gib bei Verben IMMER den deutschen INFINITIV an (machen, studieren, gehen)!"
        
        prompt = (
            f"Übersetze das spanische Wort '{wort_fuer_prompt}' ins Deutsche.\n"
            f"Satz: '{original_satz}'\n\n"
            f"Wortart: {wortart_deutsch} ({wortart_im_satz}){verb_hinweis}\n\n"
            f"Wichtige Übersetzungsregeln:\n"
            f"- Substantive: durazno→Pfirsich, manzana→Apfel, naranja→Orange, química→Chemie, pan→Brot, agua→Wasser\n"
            f"- Verben (Infinitiv): comer→essen, estudiar→studieren, hacer→machen, comprar→kaufen, beber→trinken\n"
            f"- Adjektive: rápido→schnell, grande→groß, bueno→gut, pequeño→klein\n"
            f"- Artikel: el→der, la→die, los→die, las→die, un→ein, una→eine\n"
            f"- Pronomen: él→er, ella→sie, yo→ich, nosotros→wir\n"
            f"- Präpositionen: de→von, en→in, con→mit, para→für\n\n"
            f"Antworte EXAKT in diesem Format:\n"
            f"DEUTSCH: [ein Wort]\n"
            f"KATEGORIE: [Themenbereich wie Essen, Wetter, Schule, Alltag, Körper, Familie, Kleidung, etc.]\n\n"
            f"Beispiele:\n"
            f"- durazno → DEUTSCH: Pfirsich, KATEGORIE: Essen\n"
            f"- naranja → DEUTSCH: Orange, KATEGORIE: Essen\n"
            f"- calor → DEUTSCH: Hitze, KATEGORIE: Wetter\n"
            f"- estudiar → DEUTSCH: studieren, KATEGORIE: Schule\n\n"
            f"Keine Erklärungen, nur diese 2 Zeilen!"
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
                deutsch = f"[{wort_fuer_prompt}]"
            
            # KORREKTUR: Wenn deutsches Wort großgeschrieben ist → es ist ein Substantiv!
            # (wird für Wortart-Erkennung verwendet, nicht für Kategorie)
            if deutsch and len(deutsch) > 0 and deutsch[0].isupper():
                # Deutsche Substantive sind IMMER großgeschrieben
                print(f"   [KORREKTUR] '{deutsch}' ist großgeschrieben → '{wort}' ist ein Substantiv!")
                # kategorie bleibt thematisch (vom LLM bestimmt)
            
            # Speichere Übersetzung für LLM (mit Satzform als Key)
            uebersetzungen[satzform] = deutsch
                
        except Exception as e:
            print(f"⚠️  Fehler beim Abrufen der Übersetzung für '{wort_fuer_prompt}': {e}")
            deutsch = ""
            kategorie = "Unbekannt"
        
        # Zur CSV hinzufügen - WICHTIG: Grundform/Lemma speichern!
        # Nur bei Pronomen die Satzform (wegen Akzenten: él, ella)
        zu_speicherndes_wort = wort  # Standardmäßig Lemma/Grundform
        if wortart_im_satz == "PRON":
            # Bei Pronomen: Satzform speichern (él, ella - mit Akzent)
            zu_speicherndes_wort = satzform
        
        neue_zeile = {
            "Spanisch": zu_speicherndes_wort,  # Lemma/Grundform (durazno, comer, rápido)
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
        
        print(f"   ✓ {zu_speicherndes_wort} → {deutsch} ({kategorie})")
    
    print(f"🆕 {len(neue_worter)} neue Vokabel(n) mit Übersetzung hinzugefügt.")
    
    return uebersetzungen  # Gib die Übersetzungen zurück für LLM-Satzübersetzung


if __name__ == "__main__":

    index = build_test_index()
    uebersetze_und_lerne(index,satz="Yo compro duraznos frescos",csv_datei="../vokabeln.csv")