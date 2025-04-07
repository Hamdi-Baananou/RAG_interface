# llm_interface.py
import requests
import json
from typing import List, Dict, Optional
from loguru import logger
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.docstore.document import Document

# Recommended: Use LangChain's Groq integration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config # Import configuration

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
    if not config.GROQ_API_KEY:
        logger.error("Groq API key is not configured.")
        raise ValueError("Groq API Key is missing.")

    logger.info(f"Initializing Groq LLM: {config.LLM_MODEL_NAME}")
    try:
        llm = ChatGroq(
            temperature=config.LLM_TEMPERATURE,
            model_name=config.LLM_MODEL_NAME,
            max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            # api_key=config.GROQ_API_KEY # Handled automatically if GROQ_API_KEY env var is set
        )
    except Exception as e:
         logger.error(f"Failed to initialize ChatGroq: {e}", exc_info=True)
         raise ConnectionError(f"Could not initialize Groq LLM: {e}")


    # Define the prompt template
    template = """You are a helpful assistant. Answer the user's question based *only* on the provided context chunks from PDF documents.
If the context doesn't contain the answer, state that you cannot answer based on the provided information.
When possible, mention the source document and page number (e.g., 'According to document X, page Y...') where the information was found.
Do not make up information or answer based on prior knowledge outside the context.

Context Chunks:
---------------------
{context}
---------------------

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(f"Invoking RAG chain for question: {question[:100]}...")
    try:
        answer = rag_chain.invoke(question)
        logger.success("Successfully received answer from LLM.")
        return answer.strip() if answer else "The LLM returned an empty answer."

    except Exception as e:
        # Catch potential API errors, rate limits, etc. from Langchain's execution
        logger.error(f"Error invoking RAG chain: {e}", exc_info=True)
        # You might want to inspect the exception type for specific handling
        if "authentication" in str(e).lower():
             raise PermissionError("Groq API authentication failed. Check your key.")
        elif "rate limit" in str(e).lower():
             raise ConnectionAbortedError("Groq API rate limit exceeded. Please wait.")
        elif "context length" in str(e).lower() or "too large" in str(e).lower():
             raise ValueError("Input is too large for the model's context window, even after splitting.")
        else:
             raise RuntimeError(f"Failed to get answer from LLM: {e}")


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