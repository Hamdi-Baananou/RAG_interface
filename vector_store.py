# vector_store.py
from typing import List, Optional
from loguru import logger
import os
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever
from chromadb import Client as ChromaClient

import config # Import configuration

# --- Embedding Function ---
@logger.catch(reraise=True) # Automatically log exceptions
def get_embedding_function():
    """Initializes and returns the HuggingFace embedding function."""
    model_kwargs = {'device': config.EMBEDDING_DEVICE}
    encode_kwargs = {'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings

# --- ChromaDB Setup and Retrieval ---
_chroma_client = None # Module-level client cache

def get_chroma_client():
    """Gets or creates the ChromaDB client based on config."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing Chroma client (Persistent: {config.CHROMA_SETTINGS.is_persistent})")
        if config.CHROMA_SETTINGS.is_persistent:
            logger.info(f"Chroma persistence directory: {config.CHROMA_PERSIST_DIRECTORY}")
            # Ensure directory exists if persistent
            if config.CHROMA_PERSIST_DIRECTORY and not os.path.exists(config.CHROMA_PERSIST_DIRECTORY):
                 os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        _chroma_client = ChromaClient(config.CHROMA_SETTINGS)
        logger.success("Chroma client initialized.")
    return _chroma_client

# --- Vector Store Setup ---
@logger.catch(reraise=True)
def setup_vector_store(
    documents: List[Document],
    embedding_function,
) -> Optional[VectorStoreRetriever]:
    """
    Sets up the Chroma vector store. Creates a new one if it doesn't exist,
    or potentially adds to an existing one (current logic replaces).
    Args:
        documents: List of Langchain Document objects.
        embedding_function: The embedding function to use.
    Returns:
        A VectorStoreRetriever object or None if setup fails.
    """
    if not documents:
        logger.warning("No documents provided to setup_vector_store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for setup_vector_store.")
        return None

    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    logger.info(f"Setting up vector store. Persistence directory: '{persist_directory}', Collection: '{collection_name}'")

    # Check if the directory exists, maybe Chroma needs it? (Optional check)
    # if persist_directory and not os.path.exists(persist_directory):
    #     logger.info(f"Creating persistence directory: {persist_directory}")
    #     os.makedirs(persist_directory)

    try:
        # If persisting, Chroma.from_documents handles creation and persistence directly
        # when the persist_directory argument is provided.
        logger.info(f"Creating/Updating vector store '{collection_name}' with {len(documents)} document chunks...")

        # *** Add persist_directory argument here ***
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=persist_directory # <-- This is the crucial addition
        )

        # Ensure persistence after creation/update
        if persist_directory:
            logger.info(f"Persisting vector store to directory: {persist_directory}")
            vector_store.persist() # Explicitly call persist just in case

        logger.success(f"Vector store '{collection_name}' created/updated and persisted successfully.")
        # Return the retriever
        return vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    except Exception as e:
        logger.error(f"Failed to create or populate Chroma vector store '{collection_name}': {e}", exc_info=True)
        return None

# --- Load Existing Vector Store ---
@logger.catch(reraise=True)
def load_existing_vector_store(embedding_function) -> Optional[VectorStoreRetriever]:
    """
    Loads an existing Chroma vector store from the persistent directory.
    Args:
        embedding_function: The embedding function to use.
    Returns:
        A VectorStoreRetriever object if the store exists and loads, otherwise None.
    """
    persist_directory = config.CHROMA_PERSIST_DIRECTORY
    collection_name = config.COLLECTION_NAME

    if not persist_directory:
        logger.warning("Persistence directory not configured. Cannot load existing store.")
        return None
    if not embedding_function:
        logger.error("Embedding function is not available for load_existing_vector_store.")
        return None

    if not os.path.exists(persist_directory):
         logger.warning(f"Persistence directory '{persist_directory}' does not exist. Cannot load.")
         return None

    logger.info(f"Attempting to load existing vector store from: '{persist_directory}', Collection: '{collection_name}'")

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        # Simple check to see if it loaded something (e.g., count items)
        # Note: .count() might not exist directly, use a different check if needed
        # A simple successful initialization might be enough indication
        # try:
        #     count = vector_store._collection.count() # Example internal access, might change
        #     logger.info(f"Successfully loaded collection '{collection_name}' with {count} items.")
        # except Exception:
        #      logger.warning(f"Loaded collection '{collection_name}', but could not verify item count.")

        logger.success(f"Successfully loaded vector store '{collection_name}'.")
        return vector_store.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    except Exception as e:
        # This exception block might catch cases where the collection *within* the directory doesn't exist
        # or other Chroma loading errors.
        logger.warning(f"Failed to load existing vector store '{collection_name}' from '{persist_directory}': {e}", exc_info=False) # Log less verbosely maybe
        # Log specific known issues like collection not found separately if possible
        if "does not exist" in str(e).lower(): # Basic check
             logger.warning(f"Persistent collection '{collection_name}' not found in directory '{persist_directory}'. Cannot load.")

        return None

# Add this import at the top of vector_store.py