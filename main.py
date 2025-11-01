"""
Vokabeltrainer - Hauptprogramm
Kombiniert Vokabeltest-Erstellung und Satzübersetzung mit Vokabellernen
"""

import os
import sys
from datetime import datetime

# Importiere die beiden Module
import csvparser
import translating


def zeige_banner():
    """Zeigt das Startbanner"""
    print("\n" + "="*70)
    print("🎓 VOKABELTRAINER - Spanisch ↔ Deutsch 🇪🇸🇩🇪")
    print("="*70)
    print()


def zeige_hauptmenu():
    """Zeigt das Hauptmenü und gibt die Auswahl zurück"""
    print("\n📚 HAUPTMENÜ:")
    print("  1️⃣  Vokabeltest erstellen (Quiz-Modus)")
    print("  2️⃣  Satz übersetzen & Vokabeln lernen (Lern-Modus)")
    print("  0️⃣  Beenden")
    print()
    
    while True:
        auswahl = input("Wähle eine Option (0-2): ").strip()
        if auswahl in ['0', '1', '2']:
            return auswahl
        print("❌ Ungültige Eingabe! Bitte 0, 1 oder 2 eingeben.")


def vokabeltest_modus():
    """Startet den Vokabeltest-Erstellungsmodus"""
    print("\n" + "="*70)
    print("📝 VOKABELTEST ERSTELLEN")
    print("="*70)
    
    # Lade Daten und erstelle Index
    csv_path = '../vokabelliste.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Fehler: Datei '{csv_path}' nicht gefunden!")
        input("\n⏎ Drücke Enter um fortzufahren...")
        return
    
    print("\n⏳ Lade Vokabeldaten und erstelle Index...")
    nodes = csvparser.prepare_data(csv_path)
    if not nodes:
        print("❌ Fehler: Keine Vokabeln gefunden!")
        input("\n⏎ Drücke Enter um fortzufahren...")
        return
    
    index = csvparser.build_index(nodes)
    print("✅ Index erfolgreich erstellt!")
    
    # Kategorien aus CSV lesen
    import pandas as pd
    df = pd.read_csv(csv_path, sep=';')
    kategorien = sorted(df['Kategorie'].dropna().unique().tolist())
    
    print(f"\n📂 Verfügbare Kategorien ({len(kategorien)}):")
    for i, kat in enumerate(kategorien, 1):
        print(f"   {i}. {kat}")
    
    # Kategorie auswählen
    print("\n💡 Hinweis: Gib den Namen der Kategorie ein (z.B. 'Alltag', 'Wetter', 'Bildung')")
    kategorie = input("🔍 Kategorie-Name: ").strip()
    
    if not kategorie:
        print("⚠️  Keine Kategorie angegeben!")
        input("\n⏎ Drücke Enter um fortzufahren...")
        return
    
    if kategorie not in kategorien:
        print(f"⚠️  Kategorie '{kategorie}' nicht gefunden!")
        input("\n⏎ Drücke Enter um fortzufahren...")
        return
    
    # Anzahl Fragen
    while True:
        try:
            anzahl = input("\n🔢 Wie viele Fragen? (1-50): ").strip()
            anzahl = int(anzahl)
            if 1 <= anzahl <= 50:
                break
            print("❌ Bitte eine Zahl zwischen 1 und 50 eingeben!")
        except ValueError:
            print("❌ Bitte eine gültige Zahl eingeben!")
    
    # Quiz erstellen
    print(f"\n⏳ Erstelle {anzahl} Fragen für Kategorie '{kategorie}'...")
    quiz, correct_answers = csvparser.erstelle_vokabeltest_fuer(index, kategorie_name=kategorie, anzahl_fragen=anzahl)
    
    if not quiz:
        print("❌ Fehler beim Erstellen des Quiz!")
        input("\n⏎ Drücke Enter um fortzufahren...")
        return
    
    # Quiz anzeigen
    print("\n" + "="*70)
    print("📋 DEIN VOKABELTEST:")
    print("="*70)
    print(quiz)
    
    # Lösungen anzeigen?
    loesung = input("\n👀 Lösungen anzeigen? (j/n): ").strip().lower()
    if loesung in ['j', 'ja', 'y', 'yes']:
        print("\n" + "="*70)
        print("✅ LÖSUNGEN:")
        print("="*70)
        for ans in correct_answers:
            print(f"Frage {ans['frage_nr']}: {ans['buchstabe']}) {ans['antwort']}")
    
    input("\n⏎ Drücke Enter um fortzufahren...")


def uebersetzungs_modus():
    """Startet den Übersetzungs- und Lernmodus"""
    print("\n" + "="*70)
    print("🌍 SATZ ÜBERSETZEN & VOKABELN LERNEN")
    print("="*70)
    
    csv_path = '../vokabelliste.csv'
    
    # Erstelle Index (wird von translating.py benötigt, aber nicht wirklich genutzt)
    print("\n⏳ Initialisiere Übersetzungssystem...")
    index = translating.build_test_index()
    print("✅ System bereit!")
    
    print("\n💡 Tipps:")
    print("   - Gib einen spanischen Satz ein")
    print("   - Neue Wörter werden automatisch zur Vokabelliste hinzugefügt")
    print("   - Gib 'zurück' ein um zum Hauptmenü zurückzukehren")
    
    while True:
        print("\n" + "-"*70)
        satz = input("\n🇪🇸 Spanischer Satz (oder 'zurück'): ").strip()
        
        if not satz:
            print("⚠️  Bitte einen Satz eingeben!")
            continue
        
        if satz.lower() in ['zurück', 'zurueck', 'exit', 'quit', 'q']:
            break
        
        # Übersetze und lerne
        try:
            translating.uebersetze_und_lerne(index, satz, csv_path)
        except Exception as e:
            print(f"❌ Fehler bei der Übersetzung: {e}")
        
        # Weiter oder zurück?
        print("\n" + "-"*70)
        weiter = input("\n➡️  Weiteren Satz übersetzen? (j/n): ").strip().lower()
        if weiter not in ['j', 'ja', 'y', 'yes', '']:
            break
    
    print("\n✅ Zurück zum Hauptmenü...")


def main():
    """Hauptprogramm mit interaktivem Menü"""
    zeige_banner()
    
    while True:
        auswahl = zeige_hauptmenu()
        
        if auswahl == '0':
            print("\n👋 Auf Wiedersehen! Viel Erfolg beim Lernen!")
            print("="*70)
            sys.exit(0)
        
        elif auswahl == '1':
            vokabeltest_modus()
        
        elif auswahl == '2':
            uebersetzungs_modus()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programm wurde beendet. Bis bald!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
