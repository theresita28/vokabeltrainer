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
from llama_index.embeddings.ollama import OllamaEmbedding



# --- 0. KONFIGURATION (BASE_URL ist wichtig für Ollama) ---
# Lokale Ollama-Instanz
OLLAMA_BASE_URL = "http://localhost:11434"

# 1. Das Modell zum GENERIEREN des Tests (LLM)
Settings.llm = Ollama(
    model="llama2", 
    request_timeout=120.0,
    temperature=0.1,  # Niedrige Temperatur für Fakten und strikte Formatierung
    base_url=OLLAMA_BASE_URL
)

# 2. Das Modell zum VERRKTOREN ERSTELLEN (Embedder)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
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
    system_prompt = (
        f"Du bist ein strenger Vokabeltest-Generator. Deine einzige Wissensquelle ist der bereitgestellte Kontext. "
        f"Erstelle einen Vokabeltest mit {anzahl_fragen} Fragen. "
        "Halte dich streng an dieses Format:\n"
        "1. Die Frage muss die Spanische Vokabel sein.\n"
        "2. Die Antwortmöglichkeiten müssen Ausschließlich die Deutsche Übersetzung von Vokabeln aus dem Kontext sein.\n"
        "3. Die korrekte Deutsche Übersetzung muss immer eine der Antwortmöglichkeiten sein.\n"
        "Wenn der Kontext keine ausreichenden Vokabeln liefert, antworte AUSSCHLIESSLICH mit: 'Kategorie nicht gefunden oder zu wenig Vokabeln vorhanden.'"
    )

    # Erstellen des QueryEngine mit Filter und Prompt
    query_engine = index.as_query_engine(
        filters=kategorie_filter, # <-- Der Metadaten-Filter wird hier angewendet
        similarity_top_k=20,      # Viele Chunks aus der GEFILTERTEN Liste abrufen
        system_prompt=system_prompt
    )

    # Die allgemeine Abfrage, die das LLM zur Generierung nutzt
    abfrage = f"Erstelle einen Vokabeltest mit {anzahl_fragen} Fragen zur Kategorie {kategorie_name}."
    
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
    ergebnis_gut = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Alltag")
    print(ergebnis_gut)

    print("\n" + "="*50 + "\n")

    # 4. Testen des Filters (mit nicht existierender Kategorie)
    # Sollte den Abbruch-Prompt aus dem System-Prompt auslösen
    ergebnis_schlecht = erstelle_vokabeltest_fuer(vokabel_index, kategorie_name="Chemie")
    print(ergebnis_schlecht)