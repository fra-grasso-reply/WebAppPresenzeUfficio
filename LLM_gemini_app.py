import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import io
import zipfile
import os
import re
import json
from datetime import datetime
import cv2
import pytesseract
from pytesseract import Output
import imutils
import numpy as np
import time
from typing import List, Tuple, Dict, Any

# --- Configurazione Gemini ---
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Modello sperimentale
DEBUG = True

# --- Funzioni Helper ---

@st.cache_data(show_spinner=False)
def pdf_to_images(pdf_bytes):
    """Converte i byte di un file PDF in una lista di oggetti Immagine PIL."""
    try:
        # Verifica che il file inizi con la firma PDF (%PDF-)
        if not pdf_bytes.startswith(b'%PDF-'):
            if DEBUG: print("[DEBUG] Il file non sembra essere un PDF valido (manca la firma %PDF-)")
            st.warning("Il file non sembra essere un PDF valido. Verifica il formato del file.")
            return None
            
        images = convert_from_bytes(pdf_bytes)
        if DEBUG: print(f"[DEBUG] PDF convertito in {len(images)} immagini.")
        return images
    except Exception as e:
        st.error(f"Errore durante la conversione PDF in immagine: {e}")
        st.info("Assicurati che Poppler sia installato e accessibile nel PATH di sistema.")
        return None


def correct_image_orientation(image: Image.Image) -> Image.Image:
    """
    Rileva e corregge automaticamente l'orientamento dell'immagine usando PyTesseract.
    Inoltre, ritaglia l'immagine per rimuovere lo spazio bianco in eccesso e
    migliora la qualità dell'immagine aumentando il contrasto e riducendo il rumore.
    
    Args:
        image: Oggetto immagine PIL
        
    Returns:
        Immagine PIL con orientamento corretto, ritagliata e migliorata
    """
    try:
        
        # Converti l'immagine PIL in formato OpenCV (numpy array)
        img_cv = np.array(image)
        # Converti da RGB a BGR (formato OpenCV)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Converti in RGB per pytesseract
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        if DEBUG: print("[DEBUG] Rilevamento orientamento immagine con PyTesseract...")
        
        # Rileva l'orientamento
        try:
            results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
            
            if DEBUG:
                print(f"[DEBUG] Orientamento rilevato: {results['orientation']}")
                print(f"[DEBUG] Rotazione necessaria: {results['rotate']} gradi")
                print(f"[DEBUG] Script rilevato: {results['script']}")
            
            # Se è necessaria una rotazione
            if results['rotate'] != 0:
                if DEBUG: print(f"[DEBUG] Applicazione rotazione di {results['rotate']} gradi")
                
                # Applica la rotazione usando imutils (mantiene l'intera immagine)
                rotated = imutils.rotate_bound(img_cv, angle=results["rotate"])
                
                # Usa l'immagine ruotata per il ritaglio
                img_to_crop = rotated
            else:
                if DEBUG: print("[DEBUG] Nessuna rotazione necessaria")
                # Usa l'immagine originale per il ritaglio
                img_to_crop = img_cv
            
            # RITAGLIO AUTOMATICO DELL'IMMAGINE
            if DEBUG: print("[DEBUG] Inizio ritaglio automatico dell'immagine...")
            
            # Converti in scala di grigi
            gray = cv2.cvtColor(img_to_crop, cv2.COLOR_BGR2GRAY)
            
            # Applica una soglia per ottenere un'immagine binaria
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Trova i contorni nell'immagine binaria
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Trova il contorno più grande (presumibilmente la tabella)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Ottieni il rettangolo che racchiude il contorno
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Aggiungi un piccolo margine attorno alla tabella (10 pixel)
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img_to_crop.shape[1] - x, w + 2*margin)
                h = min(img_to_crop.shape[0] - y, h + 2*margin)
                
                # Ritaglia l'immagine
                cropped = img_to_crop[y:y+h, x:x+w]
                
                if DEBUG: print(f"[DEBUG] Immagine ritagliata con dimensioni: {w}x{h}")
                
                # MIGLIORAMENTO DELLA QUALITÀ DELL'IMMAGINE
                if DEBUG: print("[DEBUG] Inizio miglioramento qualità immagine...")
                
                # Converti in scala di grigi se non lo è già
                if len(cropped.shape) == 3:
                    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                else:
                    gray_cropped = cropped
                
                # 1. Applica equalizzazione dell'istogramma per aumentare il contrasto
                equalized = cv2.equalizeHist(gray_cropped)
                
                # 2. Applica filtro bilaterale per ridurre il rumore mantenendo i bordi
                denoised = cv2.bilateralFilter(equalized, 9, 75, 75)
                
                # 3. Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) per migliorare ulteriormente il contrasto locale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
                             
                # 4. Applica una leggera nitidezza per migliorare i dettagli
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                               
                # Converti in PIL per il ritorno
                final_image = Image.fromarray(sharpened)
                
                if DEBUG: print("[DEBUG] Rotazione, ritaglio e miglioramento completati con successo")
                return final_image
            else:
                if DEBUG: print("[DEBUG] Nessun contorno trovato per il ritaglio, restituisco l'immagine ruotata")
                # Se non sono stati trovati contorni, restituisci l'immagine ruotata/originale
                return Image.fromarray(cv2.cvtColor(img_to_crop, cv2.COLOR_BGR2RGB))
                
        except Exception as osd_error:
            if DEBUG: print(f"[DEBUG] Errore durante il rilevamento dell'orientamento: {osd_error}")
            if DEBUG: print("[DEBUG] Impossibile determinare l'orientamento, l'immagine verrà utilizzata così com'è")
            return image
            
    except Exception as e:
        if DEBUG: print(f"[DEBUG] Errore durante la correzione dell'orientamento o il ritaglio: {e}")
        # In caso di errore, restituisci l'immagine originale
        return image

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

def load_few_shot_examples(examples_folder="few_shot_examples") -> List[Tuple[Image.Image, str, str]]:
    """
    Carica immagini di esempio, annotazioni JSON e descrizioni
    """
    examples = []
    
    # Ordina i file per nome
    files = sorted(os.listdir(examples_folder))
    
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            base_name = os.path.splitext(file)[0]
            json_file = os.path.join(examples_folder, base_name + ".json")
            desc_file = os.path.join(examples_folder, base_name + ".desc.txt")
            img_file = os.path.join(examples_folder, file)
            
            if os.path.exists(json_file) and os.path.exists(desc_file):
                try:
                    img = Image.open(img_file)
                    with open(json_file, 'r', encoding='utf-8') as f:
                        annotation = f.read()
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        description = f.read()
                    examples.append((img, annotation, description))
                except Exception as e:
                    if DEBUG: print(f"[DEBUG] Errore nel caricamento dell'esempio {file}: {e}")
    
    return examples

# --- Modifica: Accetta 'image' PIL invece di 'image_bytes' ---
def extract_data_with_gemini(model: genai.GenerativeModel, image: Image.Image) -> list | None:
    """
    Invia un'immagine PIL a Gemini usando genai.GenerativeModel e chiede di estrarre i dati.
    Restituisce una lista di dizionari o None in caso di errore.
    """

    examples = load_few_shot_examples()

    prompt_text = """
    Sei un esperto nell'estrazione di dati da documenti, specializzato nell'analisi di fogli presenze con scrittura manuale.
    Il tuo obiettivo è analizzare l'immagine fornita, che rappresenta una pagina di un foglio presenze ("condica de prezență") e estrarre le seguenti informazioni per ogni persona elencata nella tabella:
    1.  **Data**: Cerca una data nel formato GG/MM/AAAA nell'intestazione o in altre parti del documento. Se trovi una data, usala per tutti i record estratti da questa immagine. Se non trovi una data specifica nell'immagine, usa il valore "Data Non Trovata".
    2.  **Nome Cognome**: Il nome completo della persona.
    3.  **Ora Arrivo**: L'orario di arrivo (colonna "ORA SOSIRE" o simile), se presente. Formato HH:MM.
    4.  **Ora Partenza**: L'orario di partenza (colonna "ORA PLECARE" o simile), se presente. Formato HH:MM.

    ISTRUZIONI IMPORTANTI PER L'ANALISI DEGLI ORARI:
    - Presta particolare attenzione all'allineamento orizzontale degli orari con i nomi delle persone
    - Se noti che una firma si estende verticalmente su più righe, assicurati di non confondere questo con dati validi per la persona nella riga sottostante.
    - Quando vedi informazioni cancellate o barrate, trattale come valori nulli (null).
    - Considera che gli orari possono essere scritti in diversi formati (09:10, 9.10, 9:10, 9.10, 9 10, ecc..)
    - Se un orario è scritto accanto a un altro (es. "18:40 18:00"), considera solo il primo orario
    - Verifica che ogni orario sia associato alla persona corretta nella stessa riga
    - Se un campo è vuoto o non contiene un orario riconoscibile, lascialo come null o vuoto
    - Assicurati che ogni orario sia nella colonna corretta (arrivo o partenza)
    - Controlla attentamente le celle che contengono più numeri o annotazioni

    È di fondamentale importanza garantire la massima precisione nell'associare ogni orario alla persona corretta. 

    Ignora le colonne relative alle firme ("SEMNATURA").
    Gestisci eventuali imprecisioni dovute alla scrittura manuale degli orari al meglio delle tue capacità.
    
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
        # Prepara i contenuti per la richiesta
    if examples:
        prompt_text += "\n\nEcco alcuni esempi di fogli presenze e le relative estrazioni corrette:\n"
        contents = [prompt_text]
        
        # Aggiungi gli esempi
        for i, (example_img, example_annotation, example_desc) in enumerate(examples, 1):
            contents.append(f"\nEsempio {i}:")
            contents.append(example_img)
            contents.append(f"Descrizione esempio {i}:\n{example_desc}\n")  # Nuova linea
            contents.append(f"Estrazione corretta per l'esempio {i}:\n{example_annotation}\n")
        
        # Aggiungi l'immagine da analizzare
        contents.append("\nAnalizza questa immagine seguendo lo stesso formato degli esempi precedenti:")
        contents.append(image)
    else:
        # Se non ci sono esempi, usa il prompt standard
        contents = [prompt_text, image]
    if DEBUG: print(f"[DEBUG] prompt: {contents}")
    # --------------------------------------------------------------------

    try:
        if DEBUG: print(f"[DEBUG] Invio richiesta a Gemini (modello: {model.model_name}) per l'immagine...")
        response = model.generate_content(
            contents=contents, # Passa la lista semplice
            generation_config=genai.types.GenerationConfig(
                temperature=0.00001
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

# Configurazione della pagina
st.set_page_config(
    page_title="Image To Text Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato per migliorare l'aspetto dell'interfaccia
st.markdown("""
<style>
    /* Stile generale della pagina */
    .main {
        padding: 2rem;
        background-color: #F9F9F9;
    }
    
    /* Stile del titolo */
    h1 {
        text-align: center;
        color: #333333;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FF4B4B;
    }
    
    /* Stile del testo descrittivo */
    .stMarkdown p {
        text-align: center;
        font-size: 20px !important;
        color: #555555;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    /* Stile dell'uploader di file */
    .stFileUploader > div > label {
        font-size: 16px;
        font-weight: 500;
        color: #444444;
    }
    
    .stFileUploader > div > div {
        border: 2px dashed #CCCCCC;
        border-radius: 10px;
        padding: 2rem;
        background-color: #FFFFFF;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #FF4B4B;
    }
    
    /* Stile del bottone */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem auto;
        width: 300px;
        height: 60px;
        border-radius: 30px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stButton > button p {
        font-size: 22px !important;
        margin: 0;
    }
    
    div.stButton > button:hover {
        background-color: #E03C3C;
        color: white !important;
        box-shadow: 0 6px 8px rgba(255, 75, 75, 0.3);
        transform: translateY(-2px);
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(255, 75, 75, 0.2);
    }
    
    /* Stile della tabella dei risultati */
    .stDataFrame {
        margin-top: 2rem;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Stile del bottone di download - modificato per essere come il bottone di conversione */
    div.stDownloadButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem auto;
        width: 300px;
        height: 60px;
        border-radius: 30px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(76, 175, 80, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stDownloadButton > button p {
        font-size: 22px !important;
        margin: 0;
    }
    
    div.stDownloadButton > button:hover {
        background-color: #3d8b40;
        color: white !important;
        box-shadow: 0 6px 8px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    div.stDownloadButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header con logo e titolo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("📄 Image To Text")
    st.markdown("""
    Carica i file PDF dei fogli presenze (singoli, multipli).
    L'applicazione analizza ogni pagina ed estrae i dati, generando poi un file Excel.
    """)

# Configurazione del modello Gemini
model = configure_gemini()

# Crea un contenitore per l'uploader
with st.container():
    st.markdown("### Carica i tuoi documenti")
    uploaded_files = st.file_uploader(
        "Trascina qui i tuoi file PDF o ZIP",
        type=["pdf", "zip"],
        accept_multiple_files=True,
        help="Puoi trascinare più file PDF o un archivio ZIP contenente PDF."
    )

# Bottone di conversione
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    process_button = st.button("Convert to EXCEL")

# Contenitore per i risultati
results_container = st.container()
with results_container:
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
        
        # Mostra un'animazione di caricamento
        with st.spinner("Preparazione all'elaborazione..."):
            time.sleep(1)  # Breve pausa per mostrare l'animazione
        
        # Crea un contenitore di stato per mostrare il progresso complessivo
        with st.status("Elaborazione in corso...", expanded=True) as status:
            # Estrazione PDF da ZIP (codice invariato)
            status.update(label="🔍 Estrazione PDF da file ZIP (se presenti)...")
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
                status.update(label="❌ Nessun file PDF trovato", state="error")
                st.warning("Nessun file PDF trovato da elaborare.")
                if DEBUG: print("[DEBUG] Nessun PDF da elaborare.")
            else:
                # Informazioni sui file trovati
                status.update(label=f"📋 Trovati {len(pdf_files_to_process)} file PDF da elaborare")
                if DEBUG: print(f"[DEBUG] Inizio ciclo elaborazione per {len(pdf_files_to_process)} PDF.")
                
                # Crea solo un contenitore per il progresso totale
                progress_text = "Progresso totale"
                progress_bar = st.progress(0, text=progress_text)
                
                # Contatori per il calcolo del progresso
                total_files = len(pdf_files_to_process)
                total_pages_processed = 0
                total_pages = 0  # Sarà calcolato durante l'elaborazione
                
                # Prima scansione per contare il numero totale di pagine
                for pdf_buffer, _ in zip(pdf_files_to_process, file_names):
                    pdf_buffer.seek(0)
                    pdf_bytes = pdf_buffer.read()
                    images = pdf_to_images(pdf_bytes)
                    if images:
                        total_pages += len(images)
                    pdf_buffer.seek(0)  # Riporta il buffer all'inizio per la lettura successiva
                
                status.update(label=f"🔍 Elaborazione di {total_files} file con {total_pages} pagine totali")
                
                # Elaborazione dei file
                for i, (pdf_buffer, original_name) in enumerate(zip(pdf_files_to_process, file_names)):
                    status.update(label=f"🔍 Elaborazione file {i+1}/{total_files}: {original_name}")
                    
                    if DEBUG: print(f"\n--- [DEBUG] Elaborazione file {i+1}/{total_files}: {original_name} ---")
                    
                    pdf_buffer.seek(0)
                    pdf_bytes = pdf_buffer.read()
                    
                    if DEBUG: print("[DEBUG]   1. Conversione PDF in immagini...")
                    images = pdf_to_images(pdf_bytes)
                    if not images:
                        st.warning(f"Impossibile convertire il PDF '{original_name}' in immagini. File saltato.")
                        continue
                    
                    # Elaborazione delle pagine del file corrente
                    for page_num, image in enumerate(images):
                        # Aggiorna solo il progresso totale
                        total_progress = (total_pages_processed) / total_pages
                        total_progress_percentage = int(total_progress * 100)
                        progress_bar.progress(total_progress, text=f"Progresso totale: {total_progress_percentage}% ({total_pages_processed}/{total_pages} pagine)")
                        
                        page_identifier = f"{original_name}_Pagina_{page_num + 1}"
                        if DEBUG: print(f"\n[DEBUG]   --- Elaborazione pagina {page_num + 1} ---")
                        
                        try:
                            # Correzione dell'orientamento dell'immagine
                            if DEBUG: print(f"[DEBUG]     Correzione orientamento immagine per pagina {page_num + 1}...")
                            corrected_image = correct_image_orientation(image)
                            
                            # Aggiorna lo stato per mostrare l'attività corrente
                            status.update(label=f"🔍 Analisi della pagina {page_num+1}/{len(images)} del file {i+1}/{total_files}")
                            
                            if DEBUG: print(f"[DEBUG]     Invio immagine PIL corretta a Gemini...")
                            # Passa l'immagine corretta a Gemini
                            page_data = extract_data_with_gemini(model, corrected_image)
                            
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
                            
                            # Incrementa il contatore delle pagine elaborate
                            total_pages_processed += 1
                            
                            # Aggiorna il progresso totale dopo ogni pagina
                            total_progress = (total_pages_processed) / total_pages
                            total_progress_percentage = int(total_progress * 100)
                            progress_bar.progress(total_progress, text=f"Progresso totale: {total_progress_percentage}% ({total_pages_processed}/{total_pages} pagine)")
                            
                        except Exception as page_error:
                            st.error(f"Errore imprevisto durante l'elaborazione di {page_identifier}: {page_error}")
                            if DEBUG: print(f"[DEBUG]     ERRORE imprevisto nell'elaborazione di {page_identifier}: {page_error}")
                            # Incrementa comunque il contatore delle pagine
                            total_pages_processed += 1
                        finally:
                            try:
                                if 'image' in locals() and image: image.close()
                            except Exception as close_err:
                                if DEBUG: print(f"[DEBUG] Errore durante image.close(): {close_err}")
                            if DEBUG: print(f"[DEBUG]     Memoria immagine rilasciata per pagina {page_num + 1}.")
                
                # Aggiorna la barra di progresso al 100% quando tutto è completato
                progress_bar.progress(1.0, text=f"Progresso totale: 100% ({total_pages}/{total_pages} pagine)")
                
                # Aggiorna lo stato finale
                status.update(label="✅ Elaborazione completata con successo!", state="complete")
                
                # Rimuovi la barra di progresso dopo un breve ritardo
                time.sleep(1)
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
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                download_placeholder.download_button(
                                    label="Download Excel",
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

# Aggiungi una sezione informativa in fondo
with st.expander("ℹ️ Come funziona questa app"):
    st.markdown("""
    ### Processo di estrazione dati
    
    1. **Caricamento**: Carica i tuoi file PDF o ZIP contenenti PDF
    2. **Analisi**: L'app converte ogni pagina in un'immagine e la analizza
    3. **Estrazione**: Vengono estratti i dati dalle tabelle di presenza in suppporto dell'Intelligenza artificiale
    4. **Risultati**: I dati estratti vengono mostrati e resi disponibili per il download, in aggiunta di un riepilogo delle presenze
    
    ### Suggerimenti per risultati ottimali
    
    - Assicurati che i documenti siano scansionati chiaramente
    - I PDF dovrebbero contenere tabelle di presenze ben strutturate
    - L'app funziona meglio con documenti in cui gli orari sono chiaramente associati ai nomi
    """)
