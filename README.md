# WebAppPresenzeUfficio

Applicazione Streamlit per l'estrazione di dati da fogli presenze in formato PDF utilizzando Google Gemini AI.

## Funzionalità
- Caricamento di file PDF singoli o multipli
- Supporto per file ZIP contenenti PDF
- Estrazione automatica dei dati di presenza tramite Google Gemini AI
- Esportazione in formato Excel con riepilogo presenze

## Requisiti
- Python 3.8+
- Poppler (per pdf2image)
- Chiave API Google Gemini

## Configurazione
1. Clona il repository
2. Installa le dipendenze: `pip install -r requirements.txt`
3. Crea un file `.streamlit/secrets.toml` con la tua chiave API: