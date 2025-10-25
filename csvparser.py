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
    model="phi3:3.8b-mini-4k-instruct-q4_K_M",       # phi3:3.8b-mini-4k-instruct-q4_K_M llama2 old
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

    # DEFINITION DES STRENGEN SYSTEM-PROMPTS
    VOKABEL_TEST_PROMPT_TEMPLATE = (
             "Du bist ein strenger Vokabeltrainer. Verwende NUR die folgenden Vokabeln.\n"
            "Erstelle einen Multiple-Choice-Test mit **genau {anzahl_fragen} Fragen**, "
            "außer es stehen weniger Vokabeln zur Verfügung. "
            "Wenn weniger Vokabeln verfügbar sind, erstelle entsprechend weniger Aufgaben.\n\n"
            
            "⚠️ WICHTIG: Für **jede** Frage gelten diese Regeln strikt:\n"
            "1. Es MUSS genau **3 Antwortmöglichkeiten (A, B, C)** geben – nicht mehr, nicht weniger.\n"
            "2. Genau **eine** der Antwortmöglichkeiten ist korrekt.\n"
            "3. Die korrekte Antwort MUSS mit einem Sternchen (*) am Ende markiert werden.\n"
            "4. Alle falschen Antworten MÜSSEN ebenfalls aus den gegebenen Vokabeln stammen.\n"
            "5. Verwende **niemals Wörter außerhalb des Kontexts**.\n\n"

            "Format für jede Frage:\n"
            "Frage: Was ist die deutsche Übersetzung von <Spanisch>?\n"
            "A) ...\n"
            "B) ...\n"
            "C) ...\n"
            "Markiere die richtige Antwort mit einem Sternchen (*).\n\n"

            "KONTEXT (nur diese Vokabeln dürfen verwendet werden):\n"
            "{context_str}\n"
    )
    # Den Prompt als LlamaIndex PromptTemplate vorbereiten
    # Die Platzhalter {query_str} und {context_str} sind LlamaIndex-intern
    # und müssen enthalten sein, aber unser Haupt-Prompt kommt als system_prompt
    # Den Test-Prompt als string anpassen, um die Anzahl der Fragen einzufügen
    final_test_prompt = VOKABEL_TEST_PROMPT_TEMPLATE.format(anzahl_fragen=anzahl_fragen, context_str="{context_str}")

    # Den Response Synthesizer konfigurieren
    # response_mode="compact" ist gut, um lange Texte zu vermeiden
    # service_context wird nicht mehr direkt verwendet, Settings.llm ist der Standard
    response_synthesizer = CompactAndRefine(
        llm=Settings.llm, # Stellen Sie sicher, dass das Mistral-Modell hier verwendet wird
        verbose=True, # Hilfreich zum Debuggen
        streaming=False,
        # Hier geben wir den Prompt als System-Prompt an
        # LlamaIndex wird dies in der finalen Generierungsphase an das LLM übergeben
        text_qa_template=PromptTemplate(final_test_prompt)
    )

    # 1. RETRIEVER ERSTELLEN UND FILTER ANWENDEN
    # Wir erstellen den Retriever direkt aus dem Index und übergeben den Metadaten-Filter.
    retriever = index.as_retriever(
        filters=kategorie_filter, # <--- Der Filter wird hier angewendet
        similarity_top_k=5        #old 20, max(anzahl_fragen * 2, 10)?
    )

    # 2. RETRIEVAL TESTEN (DEBUGGING)
    # Führen Sie den Retrieval-Teil separat aus, um die Nodes zu sehen.
    retrieved_nodes = retriever.retrieve(f"Vokabeln für Test in Kategorie {kategorie_name}")

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

    # 3. QUERY ENGINE ERSTELLEN UND AUSFÜHREN
    # Wir verwenden den bereits gefilterten Retriever, um die Generierung auszuführen.
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
    abfrage = f"Erstelle {anzahl_fragen} Vokabeltestfragen zur Kategorie {kategorie_name}."
    
    try:
        response = query_engine.query(abfrage)
        return response.response
    except Exception as e:
        return f"Fehler bei der Abfrage: {e}"


# --- 4. HAUPTPROGRAMM ---
if __name__ == "__main__":
    
    # 1. Daten vorbereiten
    vokabel_nodes = prepare_data()



    # 2. Index erstellen
    vokabel_index = build_index(vokabel_nodes)
    
    # Beispiel-Vokabelliste:
    # "Alltag", "Adjektive", "Wetter", etc.
    
    # 3. Testen der Funktion (mit korrekter Kategorie)
    ergebnis_gut = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Zeit")  #Alltag/Adjektive gut
    print(ergebnis_gut)

    print("\n" + "="*50 + "\n")

    # 4. Testen des Filters (mit nicht existierender Kategorie)
    # Sollte den Abbruch-Prompt aus dem System-Prompt auslösen
    ergebnis_schlecht = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Chemie")
    print(ergebnis_schlecht)