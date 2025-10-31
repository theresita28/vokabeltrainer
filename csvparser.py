import pandas as pd
import os
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.vector_stores import (
    MetadataFilters, 
    MetadataFilter, 
    FilterOperator
)
from llama_index.llms.ollama import Ollama
#from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine


from llama_index.embeddings.huggingface import HuggingFaceEmbedding



# --- 0. KONFIGURATION (BASE_URL ist wichtig für Ollama) ---
# Lokale Ollama-Instanz
OLLAMA_BASE_URL = "http://localhost:11434"

# 1. Das Modell zum GENERIEREN des Tests (LLM)
Settings.llm = Ollama(
    model="phi3:3.8b-mini-4k-instruct-q4_K_M",       # phi3:3.8b-mini-4k-instruct-q4_K_M, llama2 old gemma:2b-instruct
    request_timeout=120.0,
    temperature=0.1,  # Niedrige Temperatur für Fakten und strikte Formatierung
    base_url=OLLAMA_BASE_URL
)

# 2. Das Modell zum VERRKTOREN ERSTELLEN (Embedder)
""" Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
) """
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", # Ein sehr guter, kleiner und schneller Embedder
    max_length=512
    # Achtung: Dies lädt das Modell lokal herunter, einmalig.
)
# --- 1. DATENVORBEREITUNG: Nodes mit Metadaten erstellen ---
def prepare_data(file_path='../vokabelliste.csv'):
    """Lädt CSV und erstellt LlamaIndex Nodes mit Kategorie als Metadaten."""
    try:
        df = pd.read_csv(file_path, sep=';') 
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden.")
        return []

    documents = []
    for index, row in df.iterrows():
        # Text, der Vektorisiert wird (alle Vokabel-Infos)
        vokabel_content = (
            f"Spanisch: {row.get('Spanisch', '')}\n"
            f"Deutsch: {row.get('Deutsch', '')}\n"
            f"Satz: {row.get('Beispielsatz', '')}"
        )
        
        # Metadaten für die strikte Filterung
        kategorie = str(row.get('Kategorie', 'Unbekannt')).strip()
        
        doc = Document(
            text=vokabel_content,
            metadata={"kategorie_filter": kategorie}
        )
        documents.append(doc)

    # Chunking-Einstellung (ca. 100 Zeichen pro Vokabelzeile ist hier ausreichend)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=150, chunk_overlap=0) 
    nodes = node_parser.get_nodes_from_documents(documents)
    
    print(f"Vokabel-Nodes erstellt: {len(nodes)}")
    return nodes

# --- 2. INDEXIERUNG: Vektoren speichern ---
def build_index(nodes):
    """Erstellt den Vektor-Index aus den Nodes."""
    if not nodes:
        print("Kann Index nicht erstellen: Keine Nodes vorhanden.")
        return None
    index = VectorStoreIndex(nodes)
    print("Vektor-Index erfolgreich erstellt und Vokabeln eingebettet.")
    return index

# --- 3. ABFRAGE: Funktion mit Metadaten-Filter ---
def erstelle_vokabeltest_fuer(index, kategorie_name: str, anzahl_fragen: int = 5):
    """
    Führt die Abfrage durch und filtert die Vektorsuche strikt nach Kategorie.
    """
    if index is None:
        return "Fehler: Index ist nicht verfügbar."
        
    print(f"\n--- Erstelle Test für Kategorie: '{kategorie_name}' ---")

    # DEFINITION DES STRIKTEN METADATEN-FILTERS
    # Die Vektorsuche wird AUF DIESE KATEGORIE BESCHRÄNKT!
    kategorie_filter = MetadataFilters(
        filters=[
            MetadataFilter(
                key="kategorie_filter", 
                value=kategorie_name, 
                operator=FilterOperator.EQ # 'EQ' = Equals (Gleichheit)
            )
        ]
    )

    # DEFINITION DES KREATIVEN SYSTEM-PROMPTS  
    VOKABEL_TEST_PROMPT_TEMPLATE = (
            "Erstelle kreative Multiple-Choice-Fragen zu spanischen Vokabeln.\n\n"
            
            "VOKABELN (Format: spanisch – deutsch):\n"
            "{context_str}\n\n"
            
            "SPANISCHE WORTLISTE (NUR diese verwenden):\n"
            "{spanisch_liste}\n\n"
            
            "REGELN:\n"
            "- Erstelle genau {anzahl_fragen} Fragen\n"
            "- Verwende NUR Wörter aus der spanischen Wortliste oben\n"
            "- Jede Frage MUSS mit 'Frage:' beginnen\n"
            "- Das spanische Wort MUSS in einfachen Anführungszeichen stehen: 'palabra'\n\n"
            
            "KREATIVE FRAGEFORMULIERUNG (wähle für jede Frage eine andere Variante):\n"
            "- 'Was bedeutet '[Wort]' auf Deutsch?'\n"
            "- 'Wie übersetzt man '[Wort]' ins Deutsche?'\n"
            "- 'Was ist die deutsche Übersetzung von '[Wort]'?'\n"
            "- 'Welche Bedeutung hat '[Wort]'?'\n"
            "- '[Wort]' bedeutet auf Deutsch...'\n\n"
            
            "ANTWORTFORMAT (STRIKT einhalten):\n"
            "- EXAKT 3 Antwortoptionen pro Frage: A), B), C)\n"
            "- NICHT mehr als 3 Optionen!\n"
            "- Eine Antwort ist die EXAKTE deutsche Übersetzung aus der Vokabelliste\n"
            "- Zwei Antworten sind falsch - erfinde plausible, thematisch passende Wörter\n"
            "- Keine zusätzlichen Zeilen, Kommentare oder Erklärungen!\n"
            "- Nach Option C) folgt eine Leerzeile, dann die nächste Frage\n\n"
            
            "BEISPIELE (beachte: NUR 3 Optionen A, B, C):\n"
            "Frage: Was bedeutet 'perro' auf Deutsch?\n"
            "A) Katze\n"
            "B) Hund\n"
            "C) Maus\n\n"
            
            "Frage: Wie übersetzt man 'rápido' ins Deutsche?\n"
            "A) schnell\n"
            "B) langsam\n"
            "C) laut\n\n"
            
            "Frage: Welche Bedeutung hat 'casa'?\n"
            "A) Straße\n"
            "B) Haus\n"
            "C) Garten\n\n"
            
            "Erstelle jetzt {anzahl_fragen} kreative Fragen:\n"
    )

    # 1. RETRIEVER ERSTELLEN UND FILTER ANWENDEN (mit größerem Top-K)
    dynamic_top_k = max(anzahl_fragen * 5, 25)
    retriever = index.as_retriever(
        filters=kategorie_filter,
        similarity_top_k=dynamic_top_k
    )

    # 2. RETRIEVAL TESTEN (DEBUGGING)
    # Führen Sie den Retrieval-Teil separat aus, um die Nodes zu sehen.
    retrieved_nodes = retriever.retrieve(f"Vokabeln für Test in Kategorie {kategorie_name}")

    # Einzigartige Vokabeln speichern: (Spanisch, Deutsch)
    unique_vokabeln = []
    for node_with_score in retrieved_nodes:
        text = node_with_score.node.text
        try:
            spanisch = text.split("Spanisch:")[1].split("\n")[0].strip()
            deutsch = text.split("Deutsch:")[1].split("\n")[0].strip()
            
            # Filter: Ignoriere Metadaten-Artefakte
            if spanisch.lower() in ["kategorie_filter", "kategorie", "metadata"]:
                continue
            if deutsch.lower() in ["kategorie_filter", "kategorie", "metadata"]:
                continue
            
            if (spanisch, deutsch) not in unique_vokabeln:
                unique_vokabeln.append((spanisch, deutsch))
        except IndexError:
            continue

    # Context-String für den Prompt (nur echte Vokabeln)
    context_str = "\n".join([f"{s} – {d}" for s, d in unique_vokabeln])

    print(f"\n--- UNIQUE VOKABELN für LLM ---")
    for s, d in unique_vokabeln:
        print(f"  {s} – {d}")
    print("--- ENDE VOKABELN ---")

    print(f"\n--- DEBUGGING: Abgerufene Nodes für Kategorie '{kategorie_name}' ---")
    if not retrieved_nodes:
        print("KEINE Nodes vom Retriever gefunden! Der Filter ist möglicherweise zu streng oder die Kategorie falsch.")
    else:
        for i, node_with_score in enumerate(retrieved_nodes[:5]): # Zeigt die ersten 5 gefundenen Nodes
            node = node_with_score.node
            print(f"  Abgerufener Node {i}:")
            print(f"    Text (Anfang): {node.text[:80]}...")
            print(f"    Metadaten: {node.metadata}")
            print(f"    Ähnlichkeitsscore: {node_with_score.score:.2f}")
    print("--- ENDE DEBUGGING ---")

    # Spanische Wortliste vorbereiten (für striktere Anweisung im Prompt)
    spanisch_liste_text = ", ".join([s for s, _ in unique_vokabeln]) or "(leer)"

    # Anzahl Fragen effektiv (maximal so viele wie Vokabeln vorhanden)
    anzahl_fragen_eff = min(anzahl_fragen, len(unique_vokabeln))

    # Den Prompt final zusammenstellen (Context kommt vom Retriever via {context_str})
    final_test_prompt = VOKABEL_TEST_PROMPT_TEMPLATE.format(
        anzahl_fragen=anzahl_fragen_eff,
        context_str="{context_str}",
        spanisch_liste=spanisch_liste_text
    )

    # Den Response Synthesizer konfigurieren (nachdem wir die Wortliste kennen)
    response_synthesizer = CompactAndRefine(
        llm=Settings.llm,
        verbose=True,
        streaming=False,
        text_qa_template=PromptTemplate(final_test_prompt)
    )

    # 3. QUERY ENGINE ERSTELLEN UND AUSFÜHREN
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    """ # Erstellen des QueryEngine mit Filter und Prompt
    query_engine = index.as_query_engine(
        filters=kategorie_filter, # <-- Der Metadaten-Filter wird hier angewendet
        similarity_top_k=20,      # Viele Chunks aus der GEFILTERTEN Liste abrufen
        response_synthesizer=response_synthesizer
    ) """

    # Die allgemeine Abfrage, die das LLM zur Generierung nutzt
    abfrage = f"Erstelle {anzahl_fragen_eff} Vokabeltestfragen zur Kategorie {kategorie_name}."
    
    try:
        response = query_engine.query(abfrage)
        raw_response = response.response
        
        # Debugging: Zeige Raw-Response
        print("\n=== DEBUG: RAW LLM RESPONSE ===")
        print(raw_response[:500])  # Erste 500 Zeichen
        print("=== END DEBUG ===\n")

        # Post-processing: Extrahiere Lösungen und entferne Sternchen aus dem Test
        final_test, correct_answers = post_process_quiz(raw_response, unique_vokabeln)

        # Gebe Test OHNE Lösungen und separate Lösungsliste zurück
        return final_test, correct_answers
    except Exception as e:
        return f"Fehler bei der Abfrage: {e}", []

def post_process_quiz(text, correct_words):
    """
    Verarbeitet den generierten Test:
    1. Extrahiert die korrekten Antworten (mit Sternchen markiert)
    2. Validiert gegen correct_words
    3. Gibt Test OHNE Sternchen + Liste der korrekten Antworten zurück
    """
    import re
    
    # Normalize: Falls kein "Frage:" verwendet wurde, konvertiere Zeilen, die mit
    # "Was bedeutet" beginnen, zu unserem Standardformat mit "Frage: "-Präfix.
    normalized_text = re.sub(r'^(\s*)(Was bedeutet)', r'\1Frage: \2', text.strip(), flags=re.MULTILINE)

    # Split bei "Frage:" aber behalte das Wort bei
    questions = re.split(r'(Frage:)', normalized_text)
    fixed_questions = []
    correct_answers = []

    frage_nr = 0
    for i in range(1, len(questions), 2):  # Paare: "Frage:" + Text
        if i+1 >= len(questions):
            break
            
        frage_text_teil = questions[i+1].strip()
        if not frage_text_teil:
            continue
        
        # Extrahiere Fragetext (bis erste Antwort)
        frage_zeilen = frage_text_teil.split("\n")
        frage_text = frage_zeilen[0].strip() if frage_zeilen else ""
        
        # VALIDIERUNG: Prüfe ob die gefragte Vokabel in correct_words ist
        # Extrahiere spanisches Wort aus Fragetext (z.B. aus "Was bedeutet 'pequeño' auf Deutsch?")
        import re as re_module
        # Erlaube sowohl einfache als auch doppelte Anführungszeichen
        spanish_word_match = re_module.search(r"[\"']([^\"']+)[\"']", frage_text)
        spanish_word = None
        expected_deutsch = None
        
        if spanish_word_match:
            spanish_word = spanish_word_match.group(1)
            # Finde die erwartete deutsche Übersetzung aus correct_words
            for s, d in correct_words:
                if s.lower() == spanish_word.lower():
                    expected_deutsch = d
                    break
            
            # Wenn das Wort nicht in correct_words ist, überspringe die Frage
            if not expected_deutsch:
                print(f"   [WARNUNG] Frage '{frage_text}' fragt nach '{spanish_word}', das NICHT in den Vokabeln ist! Überspringe Frage.")
                continue
        else:
            print(f"   [WARNUNG] Konnte kein spanisches Wort in Frage '{frage_text}' extrahieren! Überspringe Frage.")
            continue
        
        frage_nr += 1
        
        # Extrahiere Antwortoptionen (A), B), C) mit/ohne Sternchen)
        options = re.findall(r'^\s*([ABC])\)\s*(.+?)\s*$', frage_text_teil, re.MULTILINE)
        
        if not options:
            continue

        # 1. Finde markierte Antworten und säubere sie
        starred_options = []
        clean_options = []
        
        for letter, opt in options:
            # Entferne Sternchen und Whitespace
            content = re.sub(r'\*+', '', opt).strip()
            is_correct = '*' in opt
            
            if is_correct:
                starred_options.append((letter, content))
            
            clean_options.append((letter, content))

        # 2. Die KORREKTE Antwort kommt IMMER aus der CSV (expected_deutsch)
        # Suche die Option, die der expected_deutsch entspricht
        correct_letter = None
        correct_answer = expected_deutsch  # DIE KORREKTE ANTWORT AUS DER CSV!
        
        # Suche welche Option der korrekten Antwort am nächsten kommt
        best_match_letter = None
        best_match_score = 0
        
        for letter, opt in clean_options:
            opt_lower = opt.lower().strip()
            expected_lower = expected_deutsch.lower().strip()
            
            # Exakte Übereinstimmung - perfekt!
            if opt_lower == expected_lower:
                correct_letter = letter
                break
            
            # Teilübereinstimmung bei Mehrfachübersetzungen (z.B. "Zeit" in "Zeit/Wetter")
            if '/' in expected_lower:
                parts = [p.strip() for p in expected_lower.split('/')]
                if opt_lower in parts:
                    correct_letter = letter
                    correct_answer = opt  # Verwende die vom LLM gewählte Teilübersetzung
                    break
            
            # Prüfe Flexionsformen (z.B. "klein" in "kleiner")
            if expected_lower in opt_lower and len(opt_lower) - len(expected_lower) <= 2:
                # Nur wenn die Abweichung klein ist (max 2 Zeichen für Endungen wie -er, -es)
                if len(opt_lower) - len(expected_lower) > best_match_score:
                    best_match_letter = letter
                    best_match_score = len(opt_lower) - len(expected_lower)
        
        # Wenn keine exakte Übereinstimmung gefunden wurde, verwende den besten Match
        if not correct_letter and best_match_letter:
            correct_letter = best_match_letter
            print(f"   [INFO] Frage {frage_nr}: Verwende Flexionsform '{clean_options[ord(best_match_letter)-ord('A')][1]}' für '{expected_deutsch}'")
        
        # Wenn immer noch nichts gefunden wurde, ist die Frage ungültig
        if not correct_letter:
            print(f"   [FEHLER] Frage {frage_nr}: Konnte '{expected_deutsch}' in keiner Antwort finden! Überspringe Frage.")
            frage_nr -= 1  # Frage-Nummer zurücksetzen
            continue

        # 3. Duplikate entfernen
        seen = set()
        unique_options = []
        for letter, content in clean_options:
            if content not in seen:
                unique_options.append(content)  # Nur Inhalt speichern
                seen.add(content)

        # 4. MISCHE die Optionen zufällig, um die richtige Antwort nicht immer an Position A zu haben
        import random
        random.shuffle(unique_options)
        
        # 5. Finde die neue Position der korrekten Antwort nach dem Mischen
        final_options = []
        new_correct_letter = None
        for idx, content in enumerate(unique_options[:3]):  # Max 3 Optionen
            new_letter = chr(ord('A') + idx)  # A, B, C
            final_options.append((new_letter, content))
            
            # Prüfe ob dies die korrekte Antwort ist
            if content.lower().strip() == expected_deutsch.lower().strip():
                new_correct_letter = new_letter
            # Prüfe auch Teilübereinstimmungen (für "Zeit/Wetter")
            elif '/' in expected_deutsch:
                parts = [p.strip().lower() for p in expected_deutsch.split('/')]
                if content.lower().strip() in parts:
                    new_correct_letter = new_letter
            # Prüfe Flexionsformen
            elif expected_deutsch.lower().strip() in content.lower().strip() and len(content) - len(expected_deutsch) <= 2:
                new_correct_letter = new_letter
        
        # Aktualisiere correct_letter auf die neue Position nach dem Mischen
        if new_correct_letter:
            correct_letter = new_correct_letter
        else:
            print(f"   [FEHLER] Frage {frage_nr}: Korrekte Antwort nach Mischen verloren! Überspringe Frage.")
            frage_nr -= 1
            continue

        # 6. Speichere korrekte Antwort
        # 6. Speichere korrekte Antwort
        if correct_letter and correct_answer:
            correct_answers.append({
                "frage_nr": frage_nr,
                "frage_text": frage_text,
                "buchstabe": correct_letter,
                "antwort": correct_answer
            })

        # 7. Neu zusammensetzen (OHNE Sternchen, mit gemischten Optionen)
        fixed_question = f"Frage {frage_nr}: {frage_text}\n"
        for letter, content in final_options:
            fixed_question += f"{letter}) {content}\n"
        
        fixed_questions.append(fixed_question)

    final_test = "\n".join(fixed_questions)
    return final_test, correct_answers

# --- 4. HAUPTPROGRAMM ---
if __name__ == "__main__":
    
    # 1. Daten vorbereiten
    vokabel_nodes = prepare_data()



    # 2. Index erstellen
    vokabel_index = build_index(vokabel_nodes)
    
    # Beispiel-Vokabelliste:
    # "Alltag", "Adjektive", "Wetter", etc.
    
    # 3. Testen der Funktion (mit korrekter Kategorie)
    ergebnis_gut, correct_answers1 = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Adjektive")
    print("\n=== VOKABELTEST (ohne Lösungen) ===")
    print(ergebnis_gut)
    print("\n=== LÖSUNGEN (intern gespeichert) ===")
    for ans in correct_answers1:
        print(f"Frage {ans['frage_nr']}: {ans['buchstabe']}) {ans['antwort']}")

    print("\n" + "="*50 + "\n")

    # 4. Testen des Filters (mit nicht existierender Kategorie)
    ergebnis_schlecht, correct_answers2 = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Chemie")
    print("\n=== VOKABELTEST (ohne Lösungen) ===")
    print(ergebnis_schlecht)
    if correct_answers2:
        print("\n=== LÖSUNGEN (intern gespeichert) ===")
        for ans in correct_answers2:
            print(f"Frage {ans['frage_nr']}: {ans['buchstabe']}) {ans['antwort']}")