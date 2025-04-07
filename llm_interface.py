# llm_interface.py
import requests
import json
from typing import List, Dict, Optional
from loguru import logger
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document

# Recommended: Use LangChain's Groq integration
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import config # Import configuration

# --- Initialize LLM ---
@logger.catch(reraise=True) # Keep catch for unexpected errors during init
def initialize_llm():
    """Initializes and returns the Groq LLM client. No internal logging."""
    if not config.GROQ_API_KEY:
        # logger.error("GROQ_API_KEY not found.") # Remove internal logging
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")

    try:
        llm = ChatGroq(
            temperature=config.LLM_TEMPERATURE,
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL_NAME,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS
        )
        # logger.info(f"Groq LLM initialized with model: {config.LLM_MODEL_NAME}") # Remove internal logging
        return llm
    except Exception as e:
        # logger.error(f"Failed to initialize Groq LLM: {e}") # Remove internal logging
        # Re-raise a more specific error if needed, or let @logger.catch handle it
        raise ConnectionError(f"Could not initialize Groq LLM: {e}")

# --- Option 1: Using LangChain's Groq Integration (Recommended) ---

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a string for the prompt."""
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        # Add chunk index if available (from RecursiveCharacterTextSplitter with add_start_index=True)
        start_index = doc.metadata.get('start_index', None)
        chunk_info = f"Chunk {i+1}" + (f" (starts at char {start_index})" if start_index is not None else "")
        context_parts.append(
            f"{chunk_info} from '{source}' (Page {page}):\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

@logger.catch(reraise=True)
def get_answer_from_llm_langchain(question: str, retriever: VectorStoreRetriever) -> Optional[str]:
    """
    Generates an answer using Groq via LangChain, based on retrieved context.

    Args:
        question: The user's question.
        retriever: The configured vector store retriever.

    Returns:
        The generated answer string, or None if an error occurs.
    """
    # This function relies on initialize_llm() being available, but doesn't call it directly now
    # because app.py initializes the LLM and passes it to create_extraction_chain
    # We can actually remove this function if ONLY extraction is needed.
    # For now, just ensure initialize_llm exists for app.py to call.
    pass # Keep as placeholder or remove if unused


# --- Option 2: Using Raw Requests (Your original approach, adapted) ---
# Keep this if you prefer not to use langchain_groq or need fine-grained request control

# @logger.catch(reraise=True)
# def get_answer_from_llm_requests(question: str, retriever: VectorStoreRetriever) -> Optional[str]:
#     """QA using Groq API via direct requests and retrieved context."""
#     if not config.GROQ_API_KEY:
#         logger.error("Groq API key is not configured.")
#         raise ValueError("Groq API Key is missing.")
#     if not config.GROQ_API_URL:
#         logger.error("GROQ_API_URL is not configured for requests method.")
#         raise ValueError("GROQ_API_URL is missing.")

#     logger.info(f"Retrieving document chunks for question: {question[:50]}...")
#     results = retriever.invoke(question)

#     if not results:
#         logger.warning("No relevant document chunks found in ChromaDB.")
#         return "I couldn't find relevant information in the uploaded documents to answer that question."

#     logger.info(f"Retrieved {len(results)} relevant document chunks.")

#     # Constructing the context
#     context_parts = []
#     for i, doc in enumerate(results):
#         source = doc.metadata.get('source', 'Unknown')
#         page = doc.metadata.get('page', 'N/A')
#         start_index = doc.metadata.get('start_index', None)
#         chunk_info = f"Chunk {i+1}" + (f" (starts at char {start_index})" if start_index is not None else "")
#         context_parts.append(
#             f"{chunk_info} from '{source}' (Page {page}):\n{doc.page_content}"
#         )
#     context = "\n\n---\n\n".join(context_parts)

#     # Formulating the prompt for Groq
#     system_prompt = f"""You are a helpful assistant. Answer the user's question based *only* on the provided context chunks from PDF documents.
# If the context doesn't contain the answer, state that you cannot answer based on the provided information.
# When possible, mention the source document (e.g., '{results[0].metadata.get('source', 'Unknown')}') where the information was found.
# Do not make up information."""

#     user_prompt = f"""Context Chunks:
#     ---------------------
#     {context}
#     ---------------------

#     Question: {question}

#     Answer:"""

#     # Defining Groq API request payload
#     payload = {
#         "model": config.LLM_MODEL_NAME, # Use model name from config
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         "max_tokens": config.LLM_MAX_OUTPUT_TOKENS,
#         "temperature": config.LLM_TEMPERATURE,
#         "top_p": 1, # Often 1 or slightly less
#     }

#     # Headers for Groq API
#     headers = {
#         "Authorization": f"Bearer {config.GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     logger.info(f"Sending request to Groq API (model: {config.LLM_MODEL_NAME})...")
#     try:
#         response = requests.post(
#             config.GROQ_API_URL,
#             headers=headers,
#             json=payload,
#             timeout=90 # Increased timeout
#         )
#         response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

#         response_data = response.json()

#         if 'choices' not in response_data or not response_data['choices']:
#             logger.error(f"Invalid response format from Groq API: 'choices' missing or empty. Response: {response_data}")
#             raise ValueError("Received an invalid response format from the Groq API.")

#         first_choice = response_data['choices'][0]
#         if 'message' not in first_choice or 'content' not in first_choice['message']:
#             logger.error(f"Invalid response format: 'message' or 'content' missing. Choice: {first_choice}")
#             raise ValueError("Received an incomplete response from the Groq API.")

#         message_content = first_choice['message']['content']
#         cleaned_answer = message_content.strip() if message_content else "The API returned an empty answer."

#         logger.success("Groq API call successful. Returning answer.")
#         return cleaned_answer

#     except requests.exceptions.Timeout:
#         logger.error("Network Error: Request to Groq API timed out.")
#         raise TimeoutError("Network Error: Connection to Groq timed out.")
#     except requests.exceptions.HTTPError as http_err:
#         status_code = http_err.response.status_code
#         error_text = http_err.response.text[:500] # Limit error text length
#         logger.error(f"HTTP Error {status_code} contacting Groq API: {error_text}", exc_info=True)
#         if status_code == 401:
#             raise PermissionError("Groq API authentication failed (401). Check your key.")
#         elif status_code == 429:
#             raise ConnectionAbortedError("Groq API rate limit exceeded (429). Please wait.")
#         elif status_code == 413 or "too large" in error_text.lower():
#              raise ValueError("Input payload too large for Groq API (413), even after splitting.")
#         elif status_code >= 500:
#              raise ConnectionError(f"Groq API server error ({status_code}). Please try again later.")
#         else:
#              raise ConnectionError(f"Groq API request failed with status {status_code}: {error_text}")
#     except requests.exceptions.RequestException as req_err:
#         logger.error(f"Network Error contacting Groq API: {req_err}", exc_info=True)
#         raise ConnectionError(f"Network Error: Could not connect to Groq API. {req_err}")
#     except json.JSONDecodeError as json_err:
#          logger.error(f"Failed to decode JSON response from Groq API: {json_err}", exc_info=True)
#          raise ValueError(f"Invalid JSON received from Groq API: {response.text[:200]}...") # Show start of text
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during LLM request: {e}", exc_info=True)
#         raise RuntimeError(f"An unexpected error occurred: {e}")

# --- Revert back to SINGLE Extraction Chain ---
def create_extraction_chain(retriever, llm):
    """
    Creates a RAG chain specifically for running a SINGLE extraction prompt
    against retrieved context and outputting JSON.
    (This is the version from before batching was introduced)
    """
    if retriever is None or llm is None:
        logger.error("Retriever or LLM is not initialized for extraction chain.")
        return None

    # --- Use the SINGLE output JSON template ---
    template = """
Use the following pieces of retrieved context to perform the extraction task based on the reasoning steps provided in the Extraction Instructions.
Analyze the context carefully and follow the reasoning steps precisely.

Context:
{context}

Extraction Instructions:
{extraction_instructions}

---
Final Output Requirement:
Your final output MUST be a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the extracted result determined by following the Extraction Instructions, provided as a JSON string. Examples of possible values include "GF, T", "none", "NOT FOUND", "Female", "7.2", "999".
- Do NOT include any explanations, reasoning, or any text outside of this single JSON object in your response.

Example Output Format:
{{"{attribute_key}": "extracted_value"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        # Using detailed format_docs might help context
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            start_index = doc.metadata.get('start_index', None)
            chunk_info = f"Chunk {i+1}" + (f" (starts at char {start_index})" if start_index is not None else "")
            context_parts.append(
                f"{chunk_info} from '{source}' (Page {page}):\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)
        # Or simpler version: return "\n\n".join(doc.page_content for doc in docs)


    # Define the extraction chain using LCEL
    # Takes 'extraction_instructions' and 'attribute_key' as input
    extraction_chain = (
        RunnableParallel(
            # Retrieve context based on the attribute_key for better focus
            {"context": RunnablePassthrough() | (lambda x: retriever.invoke(f"Extract information about {x['attribute_key']}")) | format_docs,
             "extraction_instructions": RunnablePassthrough(),
             "attribute_key": RunnablePassthrough() }
        )
        # Assign the inputs correctly to the prompt variables
        .assign(extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
                attribute_key=lambda x: x['attribute_key']['attribute_key'])
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("SINGLE JSON Extraction RAG chain created successfully.")
    return extraction_chain

@logger.catch(reraise=True)
def run_extraction(extraction_instructions: str, attribute_key: str, extraction_chain):
    """
    Runs a specific extraction prompt/instructions through the JSON extraction RAG chain.
    (This is the version from before batching was introduced)
    """
    if not extraction_chain:
        logger.error("Extraction chain is not available.")
        return '{"error": "Extraction chain is not initialized."}'
    if not extraction_instructions:
        logger.warning("Received empty extraction instructions.")
        return '{"error": "No extraction instructions provided."}'
    if not attribute_key:
        logger.warning("Received empty attribute key.")
        return '{"error": "No attribute key provided."}'

    try:
        logger.info(f"Invoking extraction chain for key: '{attribute_key}'")
        input_data = {"extraction_instructions": extraction_instructions, "attribute_key": attribute_key}
        response = extraction_chain.invoke(input_data)
        logger.info("Extraction chain invoked successfully.")
        # Clean potential markdown ```json ... ``` tags
        if response.strip().startswith("```json"):
            response = response.strip()[7:]
            if response.endswith("```"):
                response = response[:-3]
        return response.strip()
    except Exception as e:
        error_str = str(e).lower()
        # Add specific check for rate limit errors
        if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
             logger.error(f"Rate limit hit during extraction for '{attribute_key}': {e}", exc_info=False)
             return f'{{"error": "API Rate Limit Hit for {attribute_key}"}}'
        else:
             logger.error(f"Error invoking extraction chain for '{attribute_key}': {e}", exc_info=True)
             return f'{{"error": "An error occurred during extraction: {str(e)}"}}'

# --- REMOVE BATCH FUNCTIONS ---
# def create_batch_extraction_chain(retriever, llm): ...
# def run_batch_extraction(batch_tasks_details: str, batch_extraction_chain): ...