# config.py
import os
from dotenv import load_dotenv
from chromadb.config import Settings as ChromaSettings

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
# Recommend using Langchain's Groq integration if possible
# LLM_PROVIDER = "groq" # or "requests" if using raw requests
LLM_MODEL_NAME = "mixtral-8x7b-32768" # Example: Groq offers this
# LLM_MODEL_NAME = "qwen-qwq-32b" # Your original choice via requests
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" # Needed if using raw requests

# --- Embedding Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_KWARGS = {'device': 'cpu'} # Or 'cuda' if available and setup

# --- Vector Store Configuration ---
# Option 1: In-memory (like your original code)
# CHROMA_SETTINGS = ChromaSettings(is_persistent=False)
# CHROMA_PERSIST_DIRECTORY = None # Not needed for in-memory

# Option 2: Persistent storage
CHROMA_PERSIST_DIRECTORY = "./chroma_db_prod"
CHROMA_SETTINGS = ChromaSettings(
    is_persistent=True,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    anonymized_telemetry=False # Optional: Disable telemetry
)

CHROMA_COLLECTION_NAME = "pdf_qa_prod_collection"

# --- Text Splitting Configuration ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# --- Retriever Configuration ---
RETRIEVER_SEARCH_K = 3 # Number of chunks to retrieve

# --- LLM Request Configuration ---
LLM_TEMPERATURE = 0.1
LLM_MAX_OUTPUT_TOKENS = 1024 # Adjust based on model and expected answer length

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR

# --- Validation ---
if not GROQ_API_KEY:
    # In a real app, might raise specific error or handle differently
    print("Warning: GROQ_API_KEY not found in environment variables.")