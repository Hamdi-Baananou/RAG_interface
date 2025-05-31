# config.py
import os
from dotenv import load_dotenv
# from chromadb.config import Settings as ChromaSettings # <-- REMOVE OR COMMENT OUT this import

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
# Recommend using Langchain's Groq integration if possible
# LLM_PROVIDER = "groq" # or "requests" if using raw requests
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-qwq-32b") # Reverted to a known good default
# LLM_MODEL_NAME = "qwen-qwq-32b" # Your original choice via requests
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" # Needed if using raw requests

# --- Embedding Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu") # Add this line ('cpu' is default, 'cuda' if GPU available and configured)
NORMALIZE_EMBEDDINGS = True # Add this line (Often recommended for sentence transformers)
# EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache") # Optional: Specify cache dir

# --- Vector Store Configuration ---
# Define the persistence directory (can be None for in-memory)
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db_prod") # Use consistent variable name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_qa_prod_collection") # Use the name expected by vector_store.py

# *** Calculate the is_persistent flag ***
is_persistent = bool(CHROMA_PERSIST_DIRECTORY) # True if directory is set, False otherwise

# --- Text Splitting Configuration ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", (5000)))  # Restored
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))  # Restored

# --- Retriever Configuration ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4)) # Renamed from RETRIEVER_SEARCH_K

# --- LLM Request Configuration ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1)) # Adjusted default
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 31550))

# --- Logging ---
# LOG_LEVEL = "INFO" # Can be set via environment if needed

# --- Validation ---
if not GROQ_API_KEY:
    # In a real app, might raise specific error or handle differently
    print("Warning: GROQ_API_KEY not found in environment variables.")

# --- Simplified CHROMA_SETTINGS attribute for app.py check ---
# Define a simple object or dictionary that app.py can check for is_persistent
class SimpleChromaSettings:
    # Simple placeholder class
    def __init__(self, persistent_flag):
        self.is_persistent = persistent_flag

# *** Instantiate the SimpleChromaSettings using the calculated flag ***
CHROMA_SETTINGS = SimpleChromaSettings(is_persistent) # Pass the calculated boolean here