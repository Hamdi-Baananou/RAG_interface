# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
import streamlit as st
import os
import time
from loguru import logger
import json # Import the json library

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Updated imports from llm_interface
from llm_interface import (
    initialize_llm,
    create_extraction_chain, # Revert back to this
    run_extraction         # Revert back to this
)
# Import the prompts
from extraction_prompts import (
    # Material Properties
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_PROMPT,
    GENDER_PROMPT,
    HEIGHT_MM_PROMPT,
    LENGTH_MM_PROMPT,
    WIDTH_MM_PROMPT,
    NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT,
    MECHANICAL_CODING_PROMPT,
    COLOUR_PROMPT,
    COLOUR_CODING_PROMPT,
    # Sealing & Environmental
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_PROMPT
)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PDF Auto-Extraction with Groq", # Updated title
    page_icon="📄",
    layout="wide"
)

# --- Logging Configuration ---
# Configure Loguru logger (can be more flexible than standard logging)
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO") # Example: Keep file logging if desired
# Toasts are disabled as per previous request
# Errors will still be shown via st.error where used explicitly

# --- Application State ---
# Use Streamlit's session state to hold persistent data across reruns
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'extraction_chain' not in st.session_state:
    st.session_state.extraction_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = [] # Store names of processed files
# REMOVE chat message state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# --- Global Variables / Initialization ---
# Initialize embeddings (this is relatively heavy, do it once)
@st.cache_resource
def initialize_embeddings():
    # Let exceptions from get_embedding_function propagate
    embeddings = get_embedding_function()
    return embeddings

# Initialize LLM (also potentially heavy/needs API key check)
@st.cache_resource
def initialize_llm_cached():
    # logger.info("Attempting to initialize LLM...") # Log before calling if needed
    llm_instance = initialize_llm()
    # logger.success("LLM initialized successfully.") # Log after successful call if needed
    return llm_instance

# --- Wrap the cached function call in try-except ---
embedding_function = None
llm = None

try:
    logger.info("Attempting to initialize embedding function...")
    embedding_function = initialize_embeddings()
    if embedding_function:
         logger.success("Embedding function initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not initialize embedding model. Error: {e}")
    st.stop()

try:
    logger.info("Attempting to initialize LLM...")
    llm = initialize_llm_cached()
    if llm:
        logger.success("LLM initialized successfully.")
except Exception as e:
     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
     st.error(f"Fatal Error: Could not initialize LLM. Error: {e}")
     st.stop()

# --- Check if initializations failed ---
if embedding_function is None or llm is None:
     if not st.exception: # If st.stop() wasn't called already
        st.error("Core components (Embeddings or LLM) failed to initialize. Cannot continue.")
     st.stop()


# Try loading existing vector store and create SINGLE extraction chain
if st.session_state.retriever is None and config.CHROMA_SETTINGS.is_persistent and embedding_function:
    logger.info("Attempting to load existing vector store...")
    st.session_state.retriever = load_existing_vector_store(embedding_function)
    if st.session_state.retriever:
        logger.success("Successfully loaded retriever from persistent storage.")
        st.session_state.processed_files = ["Existing data loaded from disk"]
        # Create SINGLE extraction chain
        logger.info("Creating extraction chain from loaded retriever...")
        st.session_state.extraction_chain = create_extraction_chain(st.session_state.retriever, llm) # Use single chain function
        if not st.session_state.extraction_chain:
            st.warning("Failed to create extraction chain from loaded retriever.")
    else:
        logger.warning("No existing persistent vector store found or failed to load.")

# --- UI Layout ---
persistence_enabled = config.CHROMA_SETTINGS.is_persistent
st.title("📄 PDF Auto-Extraction with Groq") # Updated title
st.markdown("Upload PDF documents, process them, and view automatically extracted information.") # Updated description
st.markdown(f"**Model:** `{config.LLM_MODEL_NAME}` | **Embeddings:** `{config.EMBEDDING_MODEL_NAME}` | **Persistence:** `{'Enabled' if persistence_enabled else 'Disabled'}`")

# Check for API Key (LLM init already does this, but maybe keep a visual warning)
if not config.GROQ_API_KEY:
    st.warning("Groq API Key not found. Please set the GROQ_API_KEY environment variable.", icon="⚠️")


# --- Sidebar for PDF Upload and Processing ---
with st.sidebar:
    st.header("1. Document Processing")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    process_button = st.button("Process Uploaded Documents", key="process_button", type="primary")

    if process_button and uploaded_files:
        if not embedding_function or not llm:
             st.error("Core components (Embeddings or LLM) failed to initialize earlier. Cannot process documents.")
        else:
            # Reset state
            st.session_state.retriever = None
            st.session_state.extraction_chain = None # Reset single chain
            st.session_state.processed_files = []

            filenames = [f.name for f in uploaded_files]
            logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
            # --- PDF Processing ---
            with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
                processed_docs = None # Initialize
                try:
                    start_time = time.time()
                    temp_dir = os.path.join(os.getcwd(), "temp_pdf_files")
                    processed_docs = process_uploaded_pdfs(uploaded_files, temp_dir)
                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                except Exception as e:
                    logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                    st.error(f"Error processing PDFs: {e}")

            # --- Vector Store Indexing ---
            if processed_docs:
                logger.info(f"Generated {len(processed_docs)} document chunks.")
                with st.spinner("Indexing documents in vector store..."):
                    try:
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")

                        if st.session_state.retriever:
                            st.session_state.processed_files = filenames # Update list
                            logger.success("Vector store setup complete. Retriever is ready.")
                            # --- Create SINGLE Extraction Chain ---
                            with st.spinner("Preparing extraction engine..."):
                                 st.session_state.extraction_chain = create_extraction_chain(st.session_state.retriever, llm)
                            if st.session_state.extraction_chain:
                                logger.success("Extraction chain created.")
                                st.success(f"Successfully processed {len(filenames)} file(s). Results below.")
                            else:
                                st.error("Failed to create extraction chain after processing.")
                        else:
                            st.error("Failed to setup vector store after processing PDFs.")
                    except Exception as e:
                         logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                         st.error(f"Error setting up vector store: {e}")
            elif not processed_docs and uploaded_files:
                st.warning("No text could be extracted or processed from the uploaded PDFs.")

    elif process_button and not uploaded_files:
        st.warning("Please upload at least one PDF file before processing.")

    # --- Display processed files status (Simplified) ---
    st.subheader("Processing Status")
    if st.session_state.extraction_chain and st.session_state.processed_files: # Check if ready for extraction results
        st.success(f"Ready. Processed: {', '.join(st.session_state.processed_files)}")
    elif persistence_enabled and st.session_state.retriever and not st.session_state.extraction_chain:
         st.warning("Loaded existing data, but failed to create extraction chain.")
    elif persistence_enabled and st.session_state.retriever:
         st.success(f"Ready. Using existing data loaded from disk.") # Assuming chain created on load
    else:
        st.info("Upload and process PDF documents to view extracted data.")


# --- Main Area for Displaying Extraction Results ---
st.header("2. Extracted Information")

if not st.session_state.extraction_chain:
    st.info("Upload and process documents using the sidebar to see extracted results here.")
else:
    # Define the prompts (attribute keys and their instructions)
    prompts_to_run = {
        # Material Properties
        "Material Filling": MATERIAL_PROMPT,
        "Material Name": MATERIAL_NAME_PROMPT,
        # Physical / Mechanical Attributes
        "Pull-to-Seat": PULL_TO_SEAT_PROMPT,
        "Gender": GENDER_PROMPT,
        "Number of Cavities": NUMBER_OF_CAVITIES_PROMPT,
        "Number of Rows": NUMBER_OF_ROWS_PROMPT,
        "Mechanical Coding": MECHANICAL_CODING_PROMPT,
        "Colour": COLOUR_PROMPT,
        "Colour Coding": COLOUR_CODING_PROMPT,
        # Sealing & Environmental
        "Working Temperature": WORKING_TEMPERATURE_PROMPT,
        "Housing Seal": HOUSING_SEAL_PROMPT,
        "Wire Seal": WIRE_SEAL_PROMPT,
        "Sealing": SEALING_PROMPT,
        "Sealing Class": SEALING_CLASS_PROMPT,
        # Terminals & Connections
        "Contact Systems": CONTACT_SYSTEMS_PROMPT,
        "Terminal Position Assurance": TERMINAL_POSITION_ASSURANCE_PROMPT,
        "Connector Position Assurance": CONNECTOR_POSITION_ASSURANCE_PROMPT,
        "Closed Cavities": CLOSED_CAVITIES_PROMPT,
        # Assembly & Type
        "Pre-Assembled": PRE_ASSEMBLED_PROMPT,
        "Type of Connector": CONNECTOR_TYPE_PROMPT,
        "Set/Kit": SET_KIT_PROMPT,
        # Specialized Attributes
        "HV Qualified": HV_QUALIFIED_PROMPT
    }

    st.info(f"Running {len(prompts_to_run)} extraction prompts individually...")

    cols = st.columns(2)
    col_index = 0
    SLEEP_INTERVAL_SECONDS = 0.5 # Adjust this value as needed (e.g., 0.5, 1, 2)

    for prompt_name, prompt_text in prompts_to_run.items():
        current_col = cols[col_index % 2]
        col_index += 1

        with current_col:
            attribute_key = prompt_name
            json_result_str = '{"error": "Extraction not run."}'

            with st.spinner(f"Extracting {prompt_name}..."):
                try:
                    start_time = time.time()
                    # Use the single run_extraction function
                    json_result_str = run_extraction(prompt_text, attribute_key, st.session_state.extraction_chain)
                    run_time = time.time() - start_time
                    logger.info(f"Extraction for '{prompt_name}' took {run_time:.2f} seconds.")

                    # --- ADD DELAY ---
                    logger.debug(f"Sleeping for {SLEEP_INTERVAL_SECONDS}s before next request...")
                    time.sleep(SLEEP_INTERVAL_SECONDS)
                    # ---------------

                except Exception as e:
                    logger.error(f"Error during extraction call for '{prompt_name}': {e}", exc_info=True)
                    st.error(f"Could not run extraction for {prompt_name}: {e}")
                    json_result_str = f'{{"error": "Exception during extraction call: {e}"}}'

            # --- Card Implementation ---
            # (The card display logic using json.loads and badge remains the same)
            with st.container(border=True):
                # ... (Parsing logic from previous step) ...
                st.markdown(f"##### {prompt_name}")
                # --- Parse the JSON result ---
                thinking_process = "Not available." # No longer expected
                json_string_to_parse = json_result_str # Parse the direct output
                final_answer_value = "Error"
                parse_error = None

                # (Keep the think block removal logic just in case LLM still includes it)
                think_start_tag = "<think>"
                think_end_tag = "</think>"
                start_index = json_result_str.find(think_start_tag)
                end_index = json_result_str.find(think_end_tag)
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    thinking_process = json_result_str[start_index + len(think_start_tag):end_index].strip()
                    json_string_to_parse = json_result_str[end_index + len(think_end_tag):].strip()
                else:
                    json_string_to_parse = json_result_str.strip()


                try:
                    # Ensure string is not empty before parsing
                    if not json_string_to_parse:
                         raise json.JSONDecodeError("No JSON content found.", "", 0)

                    parsed_json = json.loads(json_string_to_parse)
                    if isinstance(parsed_json, dict) and attribute_key in parsed_json:
                         final_answer_value = str(parsed_json[attribute_key])
                    elif isinstance(parsed_json, dict) and "error" in parsed_json:
                         # Handle specific rate limit error message
                         if "rate limit" in parsed_json['error'].lower():
                             final_answer_value = "Rate Limit Hit"
                         else:
                             final_answer_value = f"Error: {parsed_json['error']}"
                    else:
                         final_answer_value = "Key not found"
                         logger.warning(f"Key '{attribute_key}' not found in parsed JSON: {parsed_json}")

                except json.JSONDecodeError as json_err:
                    parse_error = json_err
                    final_answer_value = "Invalid JSON"
                    logger.error(f"Failed to parse JSON for '{prompt_name}'. Error: {json_err}. String attempted: '{json_string_to_parse}'")
                except Exception as parse_exc:
                    parse_error = parse_exc
                    final_answer_value = "Parsing Error"
                    logger.error(f"Unexpected error parsing result for '{prompt_name}'. Error: {parse_exc}. String attempted: '{json_string_to_parse}'")


                # --- Display Badge ---
                badge_color = "#dc3545" # Red for error
                if not parse_error and "error" not in final_answer_value.lower() and "not found" not in final_answer_value.lower() and "invalid json" not in final_answer_value.lower() and "key not found" not in final_answer_value.lower() and "rate limit" not in final_answer_value.lower():
                    badge_color = "#28a745" # Green for success
                elif "not found" in final_answer_value.lower():
                    badge_color = "#ffc107" # Yellow for "NOT FOUND"
                elif "rate limit" in final_answer_value.lower():
                    badge_color = "#6c757d" # Grey for Rate Limit

                badge_html = f'<span style="background-color: {badge_color}; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.9em; word-wrap: break-word; display: inline-block; max-width: 100%;">{final_answer_value}</span>'
                st.markdown(badge_html, unsafe_allow_html=True)

                # --- Expander for Details ---
                # (Keep expander logic as is, showing details like raw output/parsing error)
                with st.expander("Show Details"):
                     # ... (Expander content from previous step) ...
                     # Show thinking process if available (though less likely now)
                     if thinking_process != "Not available.":
                          st.markdown("**Thinking Process:**")
                          st.code(thinking_process, language=None)
                     st.markdown("**Raw JSON Output (Attempted Parse):**")
                     st.code(json_string_to_parse, language="json")
                     if parse_error:
                          st.caption(f"JSON Parsing Error: {parse_error}")
                     # Show original raw output if different
                     if json_result_str != json_string_to_parse:
                          st.markdown("**Original Raw LLM Output:**")
                          st.code(json_result_str, language=None)


    st.success("Automated extraction complete.")

# REMOVE the previous Q&A section entirely