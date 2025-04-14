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
import pandas as pd # Add pandas import
import re # Import the 're' module for regular expressions

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
# Add state for evaluation
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = [] # List to store detailed results per field
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None # Dict to store summary metrics
# Add flag to track if extraction has run for the current data
if 'extraction_performed' not in st.session_state:
    st.session_state.extraction_performed = False

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


# --- Load existing vector store or process uploads ---
# Reset evaluation state when processing new files
def reset_evaluation_state():
    st.session_state.evaluation_results = []
    st.session_state.evaluation_metrics = None
    st.session_state.extraction_performed = False # Reset the flag here too
    # Clear data editor state if it exists
    if 'gt_editor' in st.session_state:
        del st.session_state['gt_editor']

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
        # Don't reset evaluation if loading existing data, but ensure extraction hasn't run yet
        st.session_state.extraction_performed = False # Ensure flag is false on load
        pass # Don't reset evaluation results when loading
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
            # Reset state including evaluation and the extraction flag
            st.session_state.retriever = None
            st.session_state.extraction_chain = None # Reset single chain
            st.session_state.processed_files = []
            reset_evaluation_state() # Reset evaluation results AND extraction_performed flag

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
                                # Keep extraction_performed as False here, it will run in the main section
                                st.success(f"Successfully processed {len(filenames)} file(s). Evaluation below.") # Update message
                            else:
                                st.error("Failed to create extraction chain after processing.")
                                # reset_evaluation_state() called earlier is sufficient
                        else:
                            st.error("Failed to setup vector store after processing PDFs.")
                            # reset_evaluation_state() called earlier is sufficient
                    except Exception as e:
                         logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                         st.error(f"Error setting up vector store: {e}")
                         # reset_evaluation_state() called earlier is sufficient
            elif not processed_docs and uploaded_files:
                st.warning("No text could be extracted or processed from the uploaded PDFs.")
                # reset_evaluation_state() called earlier is sufficient

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
    # Ensure evaluation state is also clear if no chain
    if not st.session_state.evaluation_results and not st.session_state.extraction_performed:
         reset_evaluation_state() # Ensure reset if no chain and extraction not done
else:
    # --- Block 1: Run Extraction (if needed) ---
    if st.session_state.extraction_chain and not st.session_state.extraction_performed:
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
        SLEEP_INTERVAL_SECONDS = 0.5 # Adjust this value as needed

        extraction_results_list = [] # Temp list to build results
        extraction_successful = True # Flag to track if all extractions ran without major issues preventing state update

        for prompt_name, prompt_text in prompts_to_run.items():
            current_col = cols[col_index % 2]
            col_index += 1

            with current_col:
                attribute_key = prompt_name
                json_result_str = '{"error": "Extraction not run."}'
                run_time = 0.0
                final_answer_value = "Error"
                parse_error = None
                is_rate_limit = False

                with st.spinner(f"Extracting {prompt_name}..."):
                    try:
                        start_time = time.time()
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
                        run_time = time.time() - start_time # Record time even on error

                # --- Card Implementation ---
                with st.container(border=True):
                    st.markdown(f"##### {prompt_name}")
                    thinking_process = "Not available."
                    raw_llm_output = json_result_str # Keep original output for debugging
                    string_to_search = raw_llm_output
                    parse_error = None # Reset parse_error for this item

                    # --- Basic Cleaning: Remove <think> tags and ```json ... ``` ---
                    think_start_tag = "<think>"
                    think_end_tag = "</think>"
                    start_index = string_to_search.find(think_start_tag)
                    end_index = string_to_search.find(think_end_tag)
                    if start_index != -1 and end_index != -1 and end_index > start_index:
                        thinking_process = string_to_search[start_index + len(think_start_tag):end_index].strip()
                        string_to_search = string_to_search[end_index + len(think_end_tag):].strip()

                    if string_to_search.startswith("```json"):
                        string_to_search = string_to_search[7:]
                        if string_to_search.endswith("```"):
                            string_to_search = string_to_search[:-3]
                    string_to_search = string_to_search.strip() # General strip

                    # --- Enhanced JSON Extraction using Regex ---
                    json_string_to_parse = None
                    parsed_json = None # Initialize parsed_json
                    try:
                        # Search for the first '{...}' block, handling nested braces
                        match = re.search(r'\{.*\}', string_to_search, re.DOTALL)
                        if match:
                            potential_json = match.group(0)
                            # Attempt to parse the extracted block
                            parsed_json = json.loads(potential_json)
                            json_string_to_parse = potential_json # Store the successfully parsed part
                            logger.debug(f"Successfully parsed JSON extracted via regex for '{prompt_name}'.")
                        else:
                            # If regex fails, maybe the original string was already JSON? Try parsing it directly.
                            parsed_json = json.loads(string_to_search)
                            json_string_to_parse = string_to_search # Store the successfully parsed part
                            logger.debug(f"Successfully parsed JSON directly (no regex needed) for '{prompt_name}'.")

                        # --- Value Extraction Logic (Revised) ---
                        if isinstance(parsed_json, dict):
                            actual_keys = list(parsed_json.keys())
                            if len(actual_keys) == 1:
                                # Successfully parsed a single key-value pair JSON
                                actual_key = actual_keys[0]
                                final_answer_value = str(list(parsed_json.values())[0]) # Get the value

                                # Log a warning if the key is not what was expected, but treat as success
                                if actual_key != attribute_key:
                                    logger.warning(f"Key mismatch for '{prompt_name}'. Expected '{attribute_key}', but found '{actual_key}'. Using the value anyway.")
                                else:
                                     logger.debug(f"Correct key '{attribute_key}' found and value extracted for '{prompt_name}'.")
                                parse_error = None # Ensure parse_error is None for this successful path

                            elif "error" in parsed_json:
                                 # Handle specific error responses from the LLM if they are formatted as JSON
                                 if "rate limit" in parsed_json['error'].lower():
                                     final_answer_value = "Rate Limit Hit"
                                     is_rate_limit = True
                                     parse_error = ValueError("Rate limit hit (reported in JSON).")
                                 else:
                                     error_msg = parsed_json['error']
                                     final_answer_value = f"Error: {error_msg}"
                                     parse_error = ValueError(f"LLM returned an error in JSON: {error_msg}")
                                     logger.warning(f"LLM returned error for '{prompt_name}': {error_msg}")

                            else:
                                # Dictionary found, but not a single key-value pair or known error format
                                final_answer_value = "Unexpected JSON Format"
                                parse_error = ValueError(f"Expected single key-value JSON or error key, but got {len(actual_keys)} keys: {actual_keys}")
                                logger.warning(f"Unexpected JSON format for '{prompt_name}'. Keys: {actual_keys}. Parsed JSON: {parsed_json}")
                        else:
                            # Parsed result was not a dictionary
                            final_answer_value = "Unexpected JSON Format"
                            parse_error = TypeError(f"Expected JSON object (dict), but got {type(parsed_json)}")
                            logger.warning(f"Expected dict, but got {type(parsed_json)} for '{prompt_name}'. Parsed JSON: {parsed_json}")

                    except json.JSONDecodeError as json_err:
                        parse_error = json_err
                        final_answer_value = "Invalid JSON"
                        logger.error(f"Failed to parse JSON for '{prompt_name}'. Error: {json_err}. String attempted: '{string_to_search}'")
                        if 'potential_json' in locals() and locals().get('potential_json') != string_to_search:
                             logger.error(f"Regex extracted this substring for parsing attempt: '{locals().get('potential_json')}'")
                    # Remove specific KeyError catch as it's handled above
                    except Exception as parse_exc:
                        parse_error = parse_exc # Catch any other unexpected errors during parsing/checking
                        final_answer_value = "Parsing Error"
                        logger.error(f"Unexpected error processing result for '{prompt_name}'. Error: {parse_exc}. String attempted: '{string_to_search}'")

                    # Determine status flags (Parse error now set for more cases)
                    is_error = bool(parse_error) or "error" in final_answer_value.lower() or "invalid json" in final_answer_value.lower() or "parsing error" in final_answer_value.lower() or "unexpected json format" in final_answer_value.lower()
                    is_not_found = "not found" in final_answer_value.lower() # Explicit "NOT FOUND" from prompt
                    is_success = not is_error and not is_not_found and not is_rate_limit

                    # --- Display Badge ---
                    badge_color = "#dc3545" # Red for error (default)
                    if is_success:
                        badge_color = "#28a745" # Green for success
                    elif is_not_found:
                        badge_color = "#ffc107" # Yellow for "NOT FOUND"
                    elif is_rate_limit:
                        badge_color = "#6c757d" # Grey for Rate Limit

                    badge_html = f'<span style="background-color: {badge_color}; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.9em; word-wrap: break-word; display: inline-block; max-width: 100%;">{final_answer_value}</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)

                    # --- Store detailed result for evaluation ---
                    result_data = {
                        'Prompt Name': prompt_name,
                        'Extracted Value': final_answer_value,
                        'Ground Truth': '', # Placeholder for user input
                        'Raw Output': json_result_str,
                        'Parse Error': str(parse_error) if parse_error else None,
                        'Is Success': is_success,
                        'Is Error': is_error and not is_rate_limit and not is_not_found, # More specific error definition might be needed
                        'Is Not Found': is_not_found,
                        'Is Rate Limit': is_rate_limit,
                        'Latency (s)': round(run_time, 2),
                        'Exact Match': None, # To be calculated
                        'Case-Insensitive Match': None # To be calculated
                    }
                    extraction_results_list.append(result_data)

                    # Check if a critical error occurred that should stop us from marking extraction as done
                    if is_error and not is_rate_limit: # e.g., fundamental chain issue
                       # Optionally add more logic here if certain errors shouldn't prevent the flag
                       pass # For now, assume any error still counts as an 'attempt'

                    # --- Expander for Details ---
                    with st.expander("Show Details"):
                         if thinking_process != "Not available.":
                              st.markdown("**Thinking Process:**")
                              st.code(thinking_process, language=None)
                         # Show the string that was *actually* attempted to be parsed by json.loads
                         st.markdown("**Cleaned String / Regex Match (Attempted Parse):**")
                         display_string = json_string_to_parse if json_string_to_parse is not None else string_to_search
                         st.code(display_string, language="json")
                         if parse_error:
                              st.caption(f"Parsing/Validation Error: {parse_error}") # Renamed caption slightly
                         # Always show the original raw output if it differs significantly
                         if raw_llm_output.strip() != display_string.strip():
                              st.markdown("**Original Raw LLM Output:**")
                              st.code(raw_llm_output, language=None)

        # --- End of Extraction Loop ---
        if extraction_successful: # Only update state if the loop completed reasonably
            st.session_state.evaluation_results = extraction_results_list # Store results in session state
            st.session_state.extraction_performed = True # Set the flag HERE after successful run
            st.success("Automated extraction complete. Enter ground truth below.")
            # st.rerun() # REMOVE or COMMENT OUT this line
        else:
            # Handle case where loop might have been interrupted (optional)
            st.error("Extraction process encountered issues. Some results may be missing.")
            # Decide if partial results should be stored or flag set
            # If you still want to proceed even with errors, you might set the flag here:
            # st.session_state.extraction_performed = True


    # --- Block 2: Display Ground Truth / Metrics (if results exist) ---
    # This part now runs regardless of the extraction_performed flag, using existing results
    # It will run immediately after the extraction block finishes in the same script run (if extraction was needed)
    # Or it will run on subsequent reruns if results are already present
    if st.session_state.evaluation_results:
        st.divider()
        st.header("3. Enter Ground Truth & Evaluate")

        # Convert results to DataFrame for easier editing
        results_df = pd.DataFrame(st.session_state.evaluation_results)

        st.info("Enter the correct 'Ground Truth' value for each field below. Leave blank if the field shouldn't exist or 'NOT FOUND' is correct.")

        # Define which columns are editable
        # Make only 'Ground Truth' editable
        disabled_cols = [col for col in results_df.columns if col != 'Ground Truth']

        # Use data editor for ground truth input
        edited_df = st.data_editor(
            results_df,
            key="gt_editor", # Assign a key to access the edited state
            use_container_width=True,
            num_rows="dynamic", # Allow variable number of rows
            disabled=disabled_cols, # Disable editing for all but Ground Truth
            column_config={ # Optional: Improve display
                 "Prompt Name": st.column_config.TextColumn(width="medium"),
                 "Extracted Value": st.column_config.TextColumn(width="medium"),
                 "Ground Truth": st.column_config.TextColumn(width="medium", help="Enter the correct value here"),
                 "Is Success": st.column_config.CheckboxColumn(width="small"),
                 "Is Error": st.column_config.CheckboxColumn(width="small"),
                 "Is Not Found": st.column_config.CheckboxColumn(width="small"),
                 "Is Rate Limit": st.column_config.CheckboxColumn(width="small"),
                 "Latency (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                 "Exact Match": st.column_config.CheckboxColumn(width="small"),
                 "Case-Insensitive Match": st.column_config.CheckboxColumn(width="small"),
                 "Raw Output": None, # Hide raw output by default
                 "Parse Error": None # Hide parse error by default
            }
        )

        calculate_button = st.button("📊 Calculate Metrics", key="calc_metrics", type="primary")

        if calculate_button:
            # --- Metric Calculation Logic ---
            # Use the edited DataFrame from the data_editor's state
            final_results_list = edited_df.to_dict('records')
            total_fields = len(final_results_list)
            success_count = 0
            error_count = 0
            not_found_count = 0
            rate_limit_count = 0
            exact_match_count = 0
            case_insensitive_match_count = 0
            total_latency = 0.0
            valid_latency_count = 0 # Count fields where latency is meaningful

            for result in final_results_list:
                extracted = str(result['Extracted Value']).strip()
                ground_truth = str(result['Ground Truth']).strip()

                # Normalize "NOT FOUND" variations for comparison
                extracted_norm = "NOT FOUND" if "not found" in extracted.lower() else extracted
                gt_norm = "NOT FOUND" if "not found" in ground_truth.lower() else ground_truth
                gt_norm = "NOT FOUND" if ground_truth == "" else gt_norm # Treat empty GT as NOT FOUND

                # Calculate matches
                is_exact_match = False
                is_case_insensitive_match = False

                # Only calculate accuracy if not an error/rate limit and GT provided
                if not result['Is Error'] and not result['Is Rate Limit']:
                    if extracted_norm == gt_norm:
                        is_exact_match = True
                        is_case_insensitive_match = True # Exact implies case-insensitive
                        if gt_norm != "NOT FOUND": # Count matches only if GT wasn't NOT FOUND
                            exact_match_count += 1
                            case_insensitive_match_count += 1
                    elif extracted_norm.lower() == gt_norm.lower():
                        is_case_insensitive_match = True
                        if gt_norm != "NOT FOUND":
                             case_insensitive_match_count += 1

                result['Exact Match'] = is_exact_match
                result['Case-Insensitive Match'] = is_case_insensitive_match

                # Count outcomes
                if result['Is Success']: success_count += 1
                if result['Is Error']: error_count += 1
                if result['Is Not Found']: not_found_count += 1 # Count how many times LLM *returned* NOT FOUND
                if result['Is Rate Limit']: rate_limit_count += 1

                # Sum latency
                if isinstance(result['Latency (s)'], (int, float)):
                    total_latency += result['Latency (s)']
                    valid_latency_count += 1


            # Calculate overall metrics
            accuracy_denominator = total_fields - error_count - rate_limit_count - not_found_count # Base accuracy on successful, non-"NOT FOUND" extractions

            st.session_state.evaluation_metrics = {
                "Total Fields": total_fields,
                "Success Count": success_count,
                "Error Count": error_count,
                "Not Found Count (Extracted)": not_found_count,
                "Rate Limit Count": rate_limit_count,
                "Exact Match Count": exact_match_count,
                "Case-Insensitive Match Count": case_insensitive_match_count,
                "Accuracy Denominator": accuracy_denominator, # Fields where accuracy is meaningful
                "Success Rate (%)": (success_count / total_fields * 100) if total_fields > 0 else 0,
                "Error Rate (%)": (error_count / total_fields * 100) if total_fields > 0 else 0,
                "Not Found Rate (%)": (not_found_count / total_fields * 100) if total_fields > 0 else 0,
                "Rate Limit Rate (%)": (rate_limit_count / total_fields * 100) if total_fields > 0 else 0,
                "Exact Match Accuracy (%)": (exact_match_count / accuracy_denominator * 100) if accuracy_denominator > 0 else 0,
                "Case-Insensitive Accuracy (%)": (case_insensitive_match_count / accuracy_denominator * 100) if accuracy_denominator > 0 else 0,
                "Average Latency (s)": (total_latency / valid_latency_count) if valid_latency_count > 0 else 0,
            }

            # Update the main results list with comparison outcomes
            st.session_state.evaluation_results = final_results_list
            st.success("Metrics calculated successfully!")
            # Rerun slightly to update the display sections below
            st.rerun()

        # --- Display Metrics Section ---
        st.divider()
        st.header("4. Evaluation Metrics")

        if st.session_state.evaluation_metrics:
            metrics = st.session_state.evaluation_metrics
            st.subheader("Summary Statistics")

            m_cols = st.columns(4)
            m_cols[0].metric("Total Fields", metrics["Total Fields"])
            m_cols[1].metric("Success Rate", f"{metrics['Success Rate (%)']:.1f}%", delta=f"{metrics['Success Count']} fields", delta_color="off")
            m_cols[2].metric("Error Rate", f"{metrics['Error Rate (%)']:.1f}%", delta=f"{metrics['Error Count']} fields", delta_color="inverse" if metrics['Error Count'] > 0 else "off")
            m_cols[3].metric("Avg Latency", f"{metrics['Average Latency (s)']:.2f}s", delta=f"over {metrics['Total Fields'] - metrics['Rate Limit Count']} calls", delta_color="off")

            m_cols2 = st.columns(4)
            m_cols2[0].metric("Exact Match Acc.", f"{metrics['Exact Match Accuracy (%)']:.1f}%", help=f"Based on {metrics['Accuracy Denominator']} fields (excluding errors, rate limits, and 'NOT FOUND' results)")
            m_cols2[1].metric("Case-Insensitive Acc.", f"{metrics['Case-Insensitive Accuracy (%)']:.1f}%")
            m_cols2[2].metric("'NOT FOUND' Rate", f"{metrics['Not Found Rate (%)']:.1f}%", delta=f"{metrics['Not Found Count (Extracted)']} fields", delta_color="off")
            m_cols2[3].metric("Rate Limit Hits", f"{metrics['Rate Limit Count']}", delta_color="inverse" if metrics['Rate Limit Count'] > 0 else "off")


            st.subheader("Detailed Results")
            # Display the final results including ground truth and matches
            st.dataframe(
                pd.DataFrame(st.session_state.evaluation_results),
                use_container_width=True,
                hide_index=True,
                 column_config={ # Reuse column config for consistency
                     "Prompt Name": st.column_config.TextColumn(width="medium"),
                     "Extracted Value": st.column_config.TextColumn(width="medium"),
                     "Ground Truth": st.column_config.TextColumn(width="medium"),
                     "Is Success": st.column_config.CheckboxColumn("Success?",width="small"),
                     "Is Error": st.column_config.CheckboxColumn("Error?", width="small"),
                     "Is Not Found": st.column_config.CheckboxColumn("Not Found?", width="small"),
                     "Is Rate Limit": st.column_config.CheckboxColumn("Rate Limit?", width="small"),
                     "Latency (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                     "Exact Match": st.column_config.CheckboxColumn("Exact?", width="small"),
                     "Case-Insensitive Match": st.column_config.CheckboxColumn("Case-Ins?", width="small"),
                     "Raw Output": st.column_config.TextColumn("Raw Output", width="large"), # Show in details
                     "Parse Error": st.column_config.TextColumn("Parse Error", width="medium") # Show in details
                }
                )

        else:
            st.info("Calculate metrics after entering ground truth to see results here.")


        # --- Export Section ---
        st.divider()
        st.header("5. Export Results")

        if st.session_state.evaluation_results:
            # Prepare data for export
            export_df = pd.DataFrame(st.session_state.evaluation_results)
            export_summary = st.session_state.evaluation_metrics if st.session_state.evaluation_metrics else {}

            # Convert DataFrame to CSV
            @st.cache_data # Cache the conversion
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(export_df)

            # Convert summary dict to JSON
            json_summary_data = json.dumps(export_summary, indent=2).encode('utf-8')

            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    label="📥 Download Detailed Results (CSV)",
                    data=csv_data,
                    file_name='detailed_extraction_results.csv',
                    mime='text/csv',
                    key='download_csv'
                )
            with export_cols[1]:
                 st.download_button(
                    label="📥 Download Summary Metrics (JSON)",
                    data=json_summary_data,
                    file_name='evaluation_summary.json',
                    mime='application/json',
                    key='download_json'
                )
        else:
            st.info("Process documents and calculate metrics to enable export.")

    # --- Block 3: Handle cases where extraction ran but yielded nothing, or hasn't run ---
    elif st.session_state.extraction_chain and st.session_state.extraction_performed:
        # This condition is now reached if extraction_performed is True,
        # but evaluation_results is still empty (e.g., all extractions failed critically,
        # or the extraction_successful flag logic prevented storing results).
        st.warning("Extraction process completed, but no valid results were generated. Check logs or raw outputs if available.")
    # Optional: Add an else block here if needed for the case where extraction_chain exists but extraction_performed is False
    # (e.g., to show a "Click Process to start extraction" message, though the main extraction block usually handles this)


# REMOVE the previous Q&A section entirely (already done)