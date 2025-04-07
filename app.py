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

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Choose which LLM interface to use:
# from llm_interface import get_answer_from_llm_requests as get_llm_answer # Use raw requests
from llm_interface import get_answer_from_llm_langchain as get_llm_answer # Use LangChain integration (recommended)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PDF QA with Groq",
    page_icon="📄",
    layout="wide"
)

# --- Logging Configuration ---
# Configure Loguru logger (can be more flexible than standard logging)
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO") # Example: Keep file logging if desired
# Comment out the lines that add st.toast sinks
# logger.add(lambda msg: st.toast(msg, icon='ℹ️'), level="INFO", filter=lambda record: record["level"].name == "INFO")
# logger.add(lambda msg: st.toast(msg, icon='⚠️'), level="WARNING", filter=lambda record: record["level"].name == "WARNING")
# Errors will still be shown via st.error where used explicitly

# --- Application State ---
# Use Streamlit's session state to hold persistent data across reruns
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = [] # Store names of processed files
if 'messages' not in st.session_state:
    st.session_state.messages = [] # For chat history display

# --- Global Variables / Initialization ---
# Initialize embeddings (this is relatively heavy, do it once)
@st.cache_resource
def initialize_embeddings():
    # logger.info("Attempting to initialize embedding function...") # Move logging outside if it triggers UI
    # REMOVE the try-except block that calls st.error
    # Let exceptions from get_embedding_function propagate
    embeddings = get_embedding_function()
    # logger.success("Embedding function initialized successfully.") # Move logging outside if it triggers UI
    return embeddings

# --- Wrap the cached function call in try-except ---
embedding_function = None
try:
    logger.info("Attempting to initialize embedding function...") # Log before calling
    embedding_function = initialize_embeddings()
    if embedding_function:
         logger.success("Embedding function initialized successfully.") # Log after successful call
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
    # Display the error OUTSIDE the cached function
    st.error(f"Fatal Error: Could not initialize embedding model. Please check model name and network. Error: {e}")
    st.stop() # Stop the app if embeddings can't load

# --- Check if initialization failed ---
if embedding_function is None and not st.exception: # Check if it failed and st.stop() wasn't called immediately
     st.error("Embedding function initialization failed. Cannot continue.")
     st.stop()

# Try loading existing vector store if persistence is enabled
if st.session_state.retriever is None and config.CHROMA_SETTINGS.is_persistent and embedding_function:
    logger.info("Attempting to load existing vector store...")
    st.session_state.retriever = load_existing_vector_store(embedding_function)
    if st.session_state.retriever:
        logger.success("Successfully loaded retriever from persistent storage.")
        # You might want to store/load which files were processed for this store
        # For simplicity, we'll just indicate it's loaded.
        st.session_state.processed_files = ["Existing data loaded from disk"]
    else:
        logger.warning("No existing persistent vector store found or failed to load.")

# --- UI Layout ---
# Update persistence check to use the config value directly if CHROMA_SETTINGS is causing issues
persistence_enabled = config.CHROMA_SETTINGS.is_persistent if hasattr(config, 'CHROMA_SETTINGS') else bool(config.CHROMA_PERSIST_DIRECTORY)
st.title("📄 PDF Question Answering with Groq")
st.markdown("Upload PDF documents, process them, and ask questions based on their content.")
st.markdown(f"**Model:** `{config.LLM_MODEL_NAME}` | **Embeddings:** `{config.EMBEDDING_MODEL_NAME}` | **Persistence:** `{'Enabled' if persistence_enabled else 'Disabled'}`")

# Check for API Key
if not config.GROQ_API_KEY:
    st.warning("Groq API Key not found. Please set the GROQ_API_KEY environment variable in your `.env` file.", icon="⚠️")
    # Optionally, provide an input field, but .env is generally better practice
    # groq_api_key_input = st.text_input("Enter Groq API Key:", type="password", key="api_key_input")
    # if groq_api_key_input:
    #     config.GROQ_API_KEY = groq_api_key_input # Use with caution, temporary override
    # else:
    #     st.stop() # Stop if key is mandatory and not provided

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
        if not embedding_function:
             st.error("Embeddings failed to initialize earlier. Cannot process documents.")
        else:
            filenames = [f.name for f in uploaded_files]
            logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)}")
            with st.spinner("Processing PDFs... Loading, cleaning, splitting..."):
                try:
                    start_time = time.time()
                    # Specify a temporary directory for processing
                    temp_dir = os.path.join(os.getcwd(), "temp_pdf_files") # Create a sub-directory
                    processed_docs = process_uploaded_pdfs(uploaded_files, temp_dir)
                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")

                except Exception as e:
                    logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                    st.error(f"Error processing PDFs: {e}")
                    processed_docs = None # Ensure we don't proceed

            if processed_docs:
                logger.info(f"Generated {len(processed_docs)} document chunks.")
                with st.spinner("Indexing documents in vector store..."):
                    try:
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")

                        if st.session_state.retriever:
                            st.session_state.processed_files = filenames # Update list of processed files
                            logger.success("Vector store setup complete. Retriever is ready.")
                            st.success(f"Successfully processed {len(filenames)} file(s) ({len(processed_docs)} chunks). Ready for questions!")
                        else:
                            st.error("Failed to setup vector store after processing PDFs.")

                    except Exception as e:
                         logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                         st.error(f"Error setting up vector store: {e}")
            elif not processed_docs and uploaded_files: # Check if files were uploaded but nothing came out
                st.warning("No text could be extracted or processed from the uploaded PDFs.")
            # Clear the uploader state after processing (optional)
            # st.rerun() # Force rerun to clear uploader state if desired

    elif process_button and not uploaded_files:
        st.warning("Please upload at least one PDF file before processing.")

    # Display processed files status
    st.subheader("Processing Status")
    if st.session_state.retriever and st.session_state.processed_files:
        st.success(f"Ready to answer questions based on: {', '.join(st.session_state.processed_files)}")
    elif persistence_enabled and not st.session_state.retriever:
         st.info("No active session. Upload files or restart if persistent store exists.")
    else:
        st.info("Upload and process PDF documents to begin.")


# --- Main Area for Q&A ---
st.header("2. Ask Questions")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if retriever is ready
    if st.session_state.retriever is None:
        st.warning("Please process PDF documents first before asking questions.")
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I cannot answer questions until documents have been processed."})
        with st.chat_message("assistant"):
             st.warning("Please process PDF documents first before asking questions.")
    elif not config.GROQ_API_KEY:
         st.warning("Groq API Key is missing. Cannot contact the LLM.")
         st.session_state.messages.append({"role": "assistant", "content": "Sorry, I cannot contact the question-answering service due to a missing API key."})
         with st.chat_message("assistant"):
             st.warning("Groq API Key is missing.")
    else:
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Getting answer for question: {prompt[:100]}...")
                    start_time = time.time()
                    answer = get_llm_answer(prompt, st.session_state.retriever)
                    response_time = time.time() - start_time
                    logger.info(f"LLM response received in {response_time:.2f} seconds.")

                    full_response = answer if answer else "Sorry, I encountered an issue retrieving the answer."
                    message_placeholder.markdown(full_response)

                except (ValueError, PermissionError, ConnectionError, ConnectionAbortedError, TimeoutError, RuntimeError) as e:
                    logger.error(f"Error getting answer from LLM: {e}", exc_info=False) # Log full trace internally
                    full_response = f"Sorry, I encountered an error: {e}"
                    message_placeholder.error(full_response)
                except Exception as e:
                    logger.error(f"An unexpected error occurred during Q&A: {e}", exc_info=True)
                    full_response = f"An unexpected critical error occurred: {e}"
                    message_placeholder.error(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})