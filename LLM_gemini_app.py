import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
from PIL import Image # Assicurati che PIL.Image sia importato
import google.generativeai as genai
import io
import zipfile
import os
import re
import json
from datetime import datetime

# --- Configurazione Gemini ---
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Modello sperimentale
DEBUG = True

# --- Funzioni Helper ---

@st.cache_data(show_spinner=False)
def pdf_to_images(pdf_bytes):
    """Converte i byte di un file PDF in una lista di oggetti Immagine PIL."""
    try:
        images = convert_from_bytes(pdf_bytes)
        if DEBUG: print(f"[DEBUG] PDF convertito in {len(images)} immagini.")
        return images
    except Exception as e:
        st.error(f"Errore durante la conversione PDF in immagine: {e}")
        st.info("Assicurati che Poppler sia installato e accessibile nel PATH di sistema.")
        return None

def configure_gemini():
    """Configura l'API Gemini usando la chiave dai segreti di Streamlit e restituisce il modello."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        if DEBUG: print(f"[DEBUG] API Gemini configurata e modello '{GEMINI_MODEL_NAME}' inizializzato con successo.")
        return model
    except (FileNotFoundError, KeyError):
         st.error("Chiave API Google (GOOGLE_API_KEY) non trovata nei segreti di Streamlit.")
         st.info("Per favore, crea il file .streamlit/secrets.toml e aggiungi GOOGLE_API_KEY = 'TUA_CHIAVE'.")
         if DEBUG: print("[DEBUG] Errore FileNotFoundError/KeyError durante l'accesso a st.secrets['GOOGLE_API_KEY']")
         return None
    except Exception as e:
        st.error(f"Errore durante la configurazione di Gemini o l'inizializzazione del modello: {e}")
        if DEBUG: print(f"[DEBUG] Errore generico durante la configurazione di Gemini: {e}")
        return None

# --- Modifica: Accetta 'image' PIL invece di 'image_bytes' ---
def extract_data_with_gemini(model: genai.GenerativeModel, image: Image.Image) -> list | None:
    """
    Invia un'immagine PIL a Gemini usando genai.GenerativeModel e chiede di estrarre i dati.
    Restituisce una lista di dizionari o None in caso di errore.
    """
    if not model:
        st.error("Modello Gemini non configurato.")
        return None

    # --- Modifica: Rimosso types.Part/Blob. Passa prompt e immagine direttamente ---
    prompt_text = """
    Sei un assistente esperto nell'estrazione di dati da documenti. Analizza l'immagine fornita, che rappresenta una pagina di un foglio presenze ("condica de prezență").
    Il tuo obiettivo è estrarre le seguenti informazioni per ogni persona elencata nella tabella:
    1.  **Data**: Cerca una data nel formato GG/MM/AAAA nell'intestazione o in altre parti del documento. Se trovi una data, usala per tutti i record estratti da questa immagine. Se non trovi una data specifica nell'immagine, usa il valore "Data Non Trovata".
    2.  **Nome Cognome**: Il nome completo della persona.
    3.  **Ora Arrivo**: L'orario di arrivo (colonna "ORA SOSIRE" o simile), se presente. Formato HH:MM.
    4.  **Ora Partenza**: L'orario di partenza (colonna "ORA PLECARE" o simile), se presente. Formato HH:MM.

    Ignora le colonne relative alle firme ("SEMNATURA").
    Gestisci eventuali imprecisioni dovute alla scrittura manuale degli orari al meglio delle tue capacità. Se un orario non è leggibile o assente, lascialo vuoto o null.

    Potresti ricevere immagini ruotate di 90,180 oppure x gradi, in quel caso gestiscile adeguatamente.

    Restituisci il risultato ESCLUSIVAMENTE come una lista JSON valida. Ogni elemento della lista deve essere un oggetto JSON con le seguenti chiavi: "Data", "Nome Cognome", "Ora Arrivo", "Ora Partenza".

    Esempio di output JSON atteso:
    [
      {
        "Data": "10/04/2025",
        "Nome Cognome": "BALEANU Bogdan-Gabriel",
        "Ora Arrivo": "09:25",
        "Ora Partenza": null
      },
      {
        "Data": "10/04/2025",
        "Nome Cognome": "CALIN Gabriela Alina",
        "Ora Arrivo": "09:00",
        "Ora Partenza": "17:30"
      },
      {
        "Data": "10/04/2025",
        "Nome Cognome": "CESARO Ludovico",
        "Ora Arrivo": null,
        "Ora Partenza": null
      }
    ]

    Se non riesci a estrarre dati o l'immagine non sembra un foglio presenze valido, restituisci una lista JSON vuota: []. Non aggiungere spiegazioni o testo aggiuntivo prima o dopo la lista JSON.
    """
    # Costruisci i contenuti come lista [prompt_string, immagine_PIL]
    contents = [prompt_text, image]
    # --------------------------------------------------------------------

    try:
        if DEBUG: print(f"[DEBUG] Invio richiesta a Gemini (modello: {model.model_name}) per l'immagine...")
        response = model.generate_content(
            contents=contents, # Passa la lista semplice
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            )
            # safety_settings=...
        )

        if DEBUG: print("[DEBUG] Risposta ricevuta da Gemini.")
        response_text = ""
        try:
             response_text = response.text
        except ValueError:
             if DEBUG: print(f"[DEBUG] Impossibile accedere a response.text direttamente (possibile blocco safety?). Controllo response.parts...")
             try:
                 if response.parts: response_text = response.parts[0].text
                 else:
                      print(f"[DEBUG] Risposta bloccata o senza parti testuali. Prompt Feedback: {response.prompt_feedback}")
                      st.warning(f"La risposta di Gemini è stata bloccata o non conteneva testo. Controlla i safety settings o il prompt. Feedback: {response.prompt_feedback}")
                      return None
             except Exception as e_parts:
                  if DEBUG: print(f"[DEBUG] Errore nell'accesso a response.parts: {e_parts}")
                  st.warning(f"Errore nell'interpretare la struttura della risposta di Gemini: {e_parts}")
                  return None
        except Exception as e_text:
             if DEBUG: print(f"[DEBUG] Errore generico nell'accesso a response.text: {e_text}")
             st.warning(f"Errore generico nell'interpretare la risposta di Gemini: {e_text}")
             return None

        response_text = response_text.strip()

        if DEBUG:
             print("\n" + "="*20 + " INIZIO TESTO da GEMINI " + "="*20)
             print(response_text)
             print("="*20 + " FINE TESTO da GEMINI " + "="*20 + "\n")

        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        response_text = response_text.strip()

        if not response_text:
             if DEBUG: print("[DEBUG] Gemini ha restituito una risposta vuota dopo la pulizia.")
             return []

        try:
            extracted_data = json.loads(response_text)
            if isinstance(extracted_data, list):
                if DEBUG: print(f"[DEBUG] JSON parsato con successo. Estratti {len(extracted_data)} record.")
                return extracted_data
            else:
                if DEBUG: print(f"[DEBUG] Errore: Gemini non ha restituito una lista JSON. Tipo restituito: {type(extracted_data)}")
                st.warning(f"Gemini ha restituito dati in un formato JSON inatteso (non una lista): {response_text}")
                return None
        except json.JSONDecodeError as json_err:
            if DEBUG: print(f"[DEBUG] Errore nel parsing della risposta JSON di Gemini: {json_err}")
            if DEBUG: print(f"[DEBUG] Testo ricevuto che ha causato l'errore: {response_text}")
            st.warning(f"Impossibile interpretare la risposta di Gemini come JSON: {response_text}")
            return None

    except Exception as e:
        st.error(f"Errore durante la chiamata API a Gemini: {e}")
        if DEBUG: print(f"[DEBUG] Errore API Gemini: {e}")
        return None

def dataframe_to_excel_bytes(dfs_dict: dict[str, pd.DataFrame]) -> bytes | None:
    """Converte un dizionario di DataFrame Pandas in byte di file Excel,
       ognuno su un foglio separato.
    """
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df_to_write in dfs_dict.items():
                df_to_write.to_excel(writer, index=False, sheet_name=sheet_name)
                if DEBUG: print(f"[DEBUG] DataFrame scritto sul foglio Excel: '{sheet_name}'.")
        if DEBUG: print("[DEBUG] File Excel con più fogli generato in memoria.")
        return output.getvalue()
    except Exception as e:
        st.error(f"Errore durante la generazione del file Excel multi-foglio: {e}")
        if DEBUG: print(f"[DEBUG] Errore generazione Excel multi-foglio: {e}")
        return None

# --- Interfaccia Utente Streamlit ---

st.set_page_config(layout="wide")
st.title("📄 Image To Text")
st.markdown(f"""
Carica i file PDF dei fogli presenze (singoli, multipli).
L'applicazione analizza ogni pagina ed estrae i dati, generando poi un file Excel.

""")

model = configure_gemini()

uploaded_files = st.file_uploader(
    "Carica PDF o ZIP",
    type=["pdf", "zip"],
    accept_multiple_files=True,
    help="Puoi trascinare più file PDF, un singolo PDF."
)

process_button = st.button("Elabora File Caricati")

results_placeholder = st.empty()
download_placeholder = st.empty()

# --- Logica Principale ---
if process_button and uploaded_files:
    if not model:
        st.error("Impossibile procedere: Il modello Gemini non è configurato correttamente (controlla la chiave API nei segreti).")
    else:
        all_extracted_data = []
        pdf_files_to_process = []
        file_names = []

        if DEBUG: print("\n[DEBUG] Pulsante 'Elabora' premuto. Inizio gestione file...")
        with st.spinner("Estrazione PDF da file ZIP (se presenti)..."):
            # ... (logica gestione file invariata) ...
            for uploaded_file in uploaded_files:
                if DEBUG: print(f"[DEBUG] Controllo file: {uploaded_file.name}, Tipo: {uploaded_file.type}")
                if uploaded_file.type == "application/zip":
                    try:
                        if DEBUG: print(f"[DEBUG] Apertura ZIP: {uploaded_file.name}")
                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                            for file_info in zip_ref.infolist():
                                if file_info.filename.lower().endswith('.pdf') and not file_info.is_dir():
                                    if DEBUG: print(f"[DEBUG]   Estrazione PDF da ZIP: {file_info.filename}")
                                    pdf_bytes = zip_ref.read(file_info.filename)
                                    pdf_files_to_process.append(io.BytesIO(pdf_bytes))
                                    file_names.append(f"{uploaded_file.name} -> {file_info.filename}")
                    except zipfile.BadZipFile:
                        st.error(f"Il file '{uploaded_file.name}' non è un file ZIP valido o è corrotto.")
                    except Exception as e:
                        st.error(f"Errore nell'elaborazione del file ZIP '{uploaded_file.name}': {e}")
                elif uploaded_file.type == "application/pdf":
                    if DEBUG: print(f"[DEBUG] Aggiunta PDF diretto: {uploaded_file.name}")
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_files_to_process.append(io.BytesIO(pdf_bytes))
                    file_names.append(uploaded_file.name)


        if not pdf_files_to_process:
            st.warning("Nessun file PDF trovato da elaborare.")
            if DEBUG: print("[DEBUG] Nessun PDF da elaborare.")
        else:
            st.info(f"Trovati {len(pdf_files_to_process)} file PDF da elaborare.")
            if DEBUG: print(f"[DEBUG] Inizio ciclo elaborazione per {len(pdf_files_to_process)} PDF.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(pdf_files_to_process)

            for i, (pdf_buffer, original_name) in enumerate(zip(pdf_files_to_process, file_names)):
                current_progress = (i + 1) / total_files
                status_text.text(f"Elaborazione file {i+1}/{total_files}: {original_name}...")
                if DEBUG: print(f"\n--- [DEBUG] Elaborazione file {i+1}/{total_files}: {original_name} ---")

                pdf_buffer.seek(0)
                pdf_bytes = pdf_buffer.read()

                if DEBUG: print("[DEBUG]   1. Conversione PDF in immagini...")
                images = pdf_to_images(pdf_bytes)
                if not images:
                    st.warning(f"Impossibile convertire il PDF '{original_name}' in immagini. File saltato.")
                    continue

                for page_num, image in enumerate(images): # Ora 'image' è l'oggetto PIL
                    page_identifier = f"{original_name}_Pagina_{page_num + 1}"
                    if DEBUG: print(f"\n[DEBUG]   --- Elaborazione pagina {page_num + 1} ---")
                    try:
                        # --- Modifica: Non serve più convertire in bytes qui ---
                        # img_byte_arr = io.BytesIO()
                        # image.save(img_byte_arr, format='PNG')
                        # image_bytes = img_byte_arr.getvalue()
                        # img_byte_arr.close()
                        # -----------------------------------------------------

                        if DEBUG: print(f"[DEBUG]     Invio immagine PIL a Gemini...")
                        # --- Chiamata a Gemini ---
                        # --- Modifica: Passa l'oggetto 'image' PIL ---
                        page_data = extract_data_with_gemini(model, image)
                        # --------------------------------------------

                        if page_data is not None:
                             if isinstance(page_data, list):
                                 if DEBUG: print(f"[DEBUG]     Gemini ha restituito {len(page_data)} record per questa pagina.")
                                 for record in page_data:
                                     if isinstance(record, dict):
                                         record['File Origine'] = original_name
                                         record['Pagina PDF'] = page_num + 1
                                     else:
                                         if DEBUG: print(f"[DEBUG] Ignorato record non-dizionario restituito da Gemini: {record}")
                                 all_extracted_data.extend([rec for rec in page_data if isinstance(rec, dict)])
                             else:
                                 if DEBUG: print(f"[DEBUG] extract_data_with_gemini ha restituito un tipo inatteso: {type(page_data)}")
                        else:
                             st.warning(f"Estrazione fallita per la pagina {page_num + 1} del file {original_name}. Controlla i log o la console per dettagli.")
                             if DEBUG: print(f"[DEBUG]     Estrazione fallita (API/JSON error) per pagina {page_num + 1}.")

                    except Exception as page_error:
                        st.error(f"Errore imprevisto durante l'elaborazione di {page_identifier}: {page_error}")
                        if DEBUG: print(f"[DEBUG]     ERRORE imprevisto nell'elaborazione di {page_identifier}: {page_error}")
                    finally:
                        try:
                            if 'image' in locals() and image: image.close()
                        except Exception as close_err:
                            if DEBUG: print(f"[DEBUG] Errore durante image.close(): {close_err}")
                        if DEBUG: print(f"[DEBUG]     Memoria immagine rilasciata per pagina {page_num + 1}.")

                progress_bar.progress(current_progress)

            status_text.text("Elaborazione completata!")
            progress_bar.empty()
            if DEBUG: print("\n--- [DEBUG] Elaborazione completata ---")

            # --- Output (invariato) ---
            if all_extracted_data:
                if DEBUG: print(f"[DEBUG] Creazione DataFrame con {len(all_extracted_data)} record totali.")
                try:
                    df = pd.DataFrame(all_extracted_data)
                    cols_order = ['File Origine', 'Pagina PDF', 'Data', 'Nome Cognome', 'Ora Arrivo', 'Ora Partenza']
                    existing_cols = [col for col in cols_order if col in df.columns]
                    df = df[existing_cols]

                    # Visualizza SOLO il DataFrame principale
                    results_placeholder.dataframe(df)
                    if DEBUG: print("[DEBUG] DataFrame principale visualizzato.")

                    # --- INIZIO BLOCCO CALCOLO RIEPILOGO (SENZA VISUALIZZAZIONE STREAMLIT) ---
                    dfs_for_excel = {'Dati Estratti': df} # Inizializza con il df principale

                    if 'Nome Cognome' in df.columns and not df.empty:
                        if DEBUG: print("[DEBUG] Calcolo riepilogo presenze per persona (per Excel)...")
                        try:
                            # Conta le occorrenze per ogni 'Nome Cognome'
                            frequency_summary = df['Nome Cognome'].value_counts().reset_index()
                            # Rinomina le colonne per chiarezza
                            frequency_summary.columns = ['Nome Cognome', 'Totale Presenze Registrate']
                            # Ordina per nome (opzionale)
                            frequency_summary = frequency_summary.sort_values(by='Nome Cognome').reset_index(drop=True)

                            # !!! RIMOSSA LA VISUALIZZAZIONE IN STREAMLIT !!!
                            # st.subheader("📊 Riepilogo Presenze Totali per Persona")
                            # st.dataframe(frequency_summary)

                            # Aggiungi il riepilogo al dizionario per Excel
                            dfs_for_excel['Riepilogo Presenze'] = frequency_summary
                            if DEBUG: print("[DEBUG] Riepilogo presenze calcolato e aggiunto per Excel.")

                        except Exception as freq_error:
                             # Se c'è un errore nel riepilogo, loggalo ma procedi con l'export del solo df principale
                             st.warning(f"Attenzione: Errore durante il calcolo del riepilogo frequenze per Excel: {freq_error}")
                             if DEBUG: print(f"[DEBUG] Errore calcolo riepilogo per Excel: {freq_error}")
                             # dfs_for_excel conterrà solo 'Dati Estratti' in questo caso

                    else:
                        if DEBUG: print("[DEBUG] Colonna 'Nome Cognome' non trovata o DataFrame vuoto, riepilogo per Excel saltato.")
                    # --- FINE BLOCCO CALCOLO RIEPILOGO ---


                    if DEBUG: print("[DEBUG] Generazione file Excel multi-foglio...")
                    # Chiama la funzione aggiornata con il dizionario
                    excel_bytes = dataframe_to_excel_bytes(dfs_for_excel) # Passa il dizionario

                    if excel_bytes:
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Aggiorna nome file e etichetta per riflettere il contenuto Excel
                        file_name = f"Report_Presenze_{current_time}.xlsx"
                        download_placeholder.download_button(
                            # Etichetta suggerita per chiarezza
                            label="📥 Scarica Report Excel (Dati + Riepilogo)",
                            data=excel_bytes,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        if DEBUG: print(f"[DEBUG] Pulsante download generato per {file_name}.")
                    else:
                         download_placeholder.empty()

                except Exception as df_excel_error:
                     st.error(f"Errore durante la creazione del DataFrame o del file Excel: {df_excel_error}")
                     if DEBUG: print(f"[DEBUG] Errore DataFrame/Excel: {df_excel_error}")
                     results_placeholder.empty()
                     download_placeholder.empty()

            else:
                results_placeholder.warning("Nessun dato è stato estratto dai file forniti")
                if DEBUG: print("[DEBUG] Nessun dato estratto, nessun output generato.")
                download_placeholder.empty()


elif process_button and not uploaded_files:
    st.warning("Per favore, carica almeno un file PDF o ZIP.")
    if DEBUG: print("[DEBUG] Pulsante premuto ma nessun file caricato.")
