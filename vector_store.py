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
    logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
    # Ensure model_kwargs uses the configured device
    model_kwargs = {'device': config.EMBEDDING_DEVICE}
    # You can add other encode_kwargs if needed, e.g., {'normalize_embeddings': True}
    encode_kwargs = {'normalize_embeddings': config.NORMALIZE_EMBEDDINGS}

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # cache_folder=config.EMBEDDING_CACHE_DIR # Optional: if you want to specify a cache dir
    )
    logger.success("Embedding function initialized successfully.")
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

@logger.catch(reraise=True)
def setup_vector_store(documents: List[Document], embeddings) -> Optional[VectorStoreRetriever]:
    """
    Sets up the Chroma vector store with the provided documents.
    Deletes the collection if it already exists before adding new documents.

    Args:
        documents: List of split Document objects.
        embeddings: The embedding function to use.

    Returns:
        A Langchain VectorStoreRetriever instance or None if setup fails.
    """
    if not documents:
        logger.warning("No documents provided to setup_vector_store. Skipping setup.")
        return None

    client = get_chroma_client()
    collection_name = config.CHROMA_COLLECTION_NAME

    # Check if collection exists and delete it (ensures fresh start)
    try:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            logger.warning(f"Collection '{collection_name}' already exists. Deleting it...")
            client.delete_collection(name=collection_name)
            logger.success(f"Collection '{collection_name}' deleted.")
        else:
             logger.info(f"Collection '{collection_name}' does not exist yet.")

    except Exception as e:
        logger.error(f"Error managing existing collection '{collection_name}': {e}", exc_info=True)
        # Depending on requirements, you might want to stop or try to continue
        # For this example, we'll try to proceed assuming deletion is best effort

    # Create the vector store using Langchain's Chroma helper
    logger.info(f"Creating new vector store '{collection_name}' with {len(documents)} document chunks...")
    try:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            client=client,
            # persist_directory is handled by the client settings now if persistent
            # persist_directory=config.CHROMA_PERSIST_DIRECTORY if config.CHROMA_SETTINGS.is_persistent else None
        )
        if config.CHROMA_SETTINGS.is_persistent:
             logger.info("Persisting vector store...")
             vector_store.persist() # Explicitly persist if using Langchain wrapper with persistent client

        logger.success(f"Successfully added {len(documents)} chunks to collection '{collection_name}'.")

        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.RETRIEVER_SEARCH_K}
        )
        logger.success(f"Chroma retriever created with k={config.RETRIEVER_SEARCH_K}.")
        return retriever

    except Exception as e:
        logger.error(f"Failed to create or populate Chroma vector store '{collection_name}': {e}", exc_info=True)
        return None

@logger.catch(reraise=True)
def load_existing_vector_store(embeddings) -> Optional[VectorStoreRetriever]:
    """
    Loads an existing persistent Chroma vector store.

    Args:
        embeddings: The embedding function to use.

    Returns:
        A Langchain VectorStoreRetriever instance or None if store doesn't exist or fails to load.
    """
    if not config.CHROMA_SETTINGS.is_persistent or not config.CHROMA_PERSIST_DIRECTORY:
        logger.warning("Cannot load existing store: Persistence is not enabled in config.")
        return None

    client = get_chroma_client()
    collection_name = config.CHROMA_COLLECTION_NAME

    # Check if collection exists
    try:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name not in existing_collections:
            logger.warning(f"Persistent collection '{collection_name}' not found in directory '{config.CHROMA_PERSIST_DIRECTORY}'. Cannot load.")
            return None
        logger.info(f"Found existing collection '{collection_name}'. Loading...")

    except Exception as e:
         logger.error(f"Error checking for existing collection '{collection_name}': {e}", exc_info=True)
         return None # Can't be sure if it exists

    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.RETRIEVER_SEARCH_K}
        )
        logger.success(f"Successfully loaded existing vector store '{collection_name}'.")
        return retriever

    except Exception as e:
        logger.error(f"Failed to load existing Chroma vector store '{collection_name}': {e}", exc_info=True)
        return None

# Add this import at the top of vector_store.py
