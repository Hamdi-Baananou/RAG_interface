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
import asyncio # Need asyncio for crawl4ai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from bs4 import BeautifulSoup # Import BeautifulSoup

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
    # Keep detailed formatting as it might help LLM locate info in PDFs
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        start_index = doc.metadata.get('start_index', None)
        chunk_info = f"Chunk {i+1}" + (f" (starts at char {start_index})" if start_index is not None else "")
        context_parts.append(
            f"{chunk_info} from '{source}' (Page {page}):\\n{doc.page_content}"
        )
    return "\\n\\n---\\n\\n".join(context_parts)

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

# --- LLM-Free Web Scraping Configuration (Revised for Table HTML) ---

# Configure websites to scrape, in order of preference.
# We now target the main table/container holding the product features.
WEBSITE_CONFIGS = [
    {
        "name": "TE Connectivity",
        "base_url_template": "https://www.te.com/en/product-{part_number}.html",
        # JS to click the features expander button if it's not already expanded
        "pre_extraction_js": (
            "(async () => {"
            "    const expandButtonSelector = '#pdp-features-expander-btn';"
            "    const featuresPanelSelector = '#pdp-features-tabpanel';"
            "    const expandButton = document.querySelector(expandButtonSelector);"
            "    const featuresPanel = document.querySelector(featuresPanelSelector);"
            "    if (expandButton && expandButton.getAttribute('aria-selected') === 'false') {"
            "        console.log('Features expand button indicates collapsed state, clicking...');"
            "        expandButton.click();"
            "        await new Promise(r => setTimeout(r, 1500));"
            "        console.log('Expand button clicked and waited.');"
            "    } else if (expandButton) {"
            "        console.log('Features expand button already indicates expanded state.');"
            "    } else {"
            "        console.log('Features expand button selector not found:', expandButtonSelector);"
            "        if (featuresPanel && !featuresPanel.offsetParent) {"
            "           console.warn('Button not found, but panel seems hidden. JS might need adjustment.');"
            "        } else if (!featuresPanel) {"
            "           console.warn('Neither expand button nor features panel found.');"
            "        }"
            "    }"
            "})();"
        ),
        # Selector for the main container holding the features/specifications table
        "table_selector": "#pdp-features-tabpanel" # Example selector - VERIFY!
    },
    {
        "name": "TraceParts",
        "base_url_template": "https://www.traceparts.com/en/search?CatalogPath=&KeepFilters=true&Keywords={part_number}&SearchAction=Keywords",
        "pre_extraction_js": None, # Assuming no interaction needed for TraceParts search results page
        # Selector for the table or div containing technical data on TraceParts
        "table_selector": ".technical-data" # Example selector - VERIFY!
    },
    # Add other supplier websites here following the same structure
]

# --- HTML Cleaning Function ---
def clean_scraped_html(html_content: str, site_name: str) -> Optional[str]:
    """
    Parses scraped HTML using BeautifulSoup and extracts key-value pairs
    from known structures (e.g., TE Connectivity feature lists).

    Args:
        html_content: The raw HTML string scraped from the website.
        site_name: The name of the site (e.g., "TE Connectivity") to apply specific parsing logic.

    Returns:
        A cleaned string representation (e.g., "Key: Value\\nKey: Value") or None if parsing fails.
    """
    if not html_content:
        return None

    logger.debug(f"Cleaning HTML content from {site_name}...")
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_texts = []

    try:
        # --- Add site-specific parsing logic here --- 
        if site_name == "TE Connectivity":
            # Find all feature list items within the main panel
            feature_items = soup.find_all('li', class_='product-feature')
            if not feature_items:
                 # Maybe the main selector was wrong? Try finding the panel first
                 panel = soup.find(id='pdp-features-tabpanel')
                 if panel:
                      feature_items = panel.find_all('li', class_='product-feature')
                 
            if feature_items:
                for item in feature_items:
                    title_span = item.find('span', class_='feature-title')
                    value_em = item.find('em', class_='feature-value')
                    if title_span and value_em:
                        title = title_span.get_text(strip=True).replace(':', '').strip()
                        value = value_em.get_text(strip=True)
                        if title and value:
                            extracted_texts.append(f"{title}: {value}")
                logger.info(f"Extracted {len(extracted_texts)} features from TE Connectivity HTML.")
            else:
                 logger.warning(f"Could not find 'li.product-feature' items in the TE Connectivity HTML provided.")

        elif site_name == "TraceParts":
            # Add parsing logic specific to TraceParts HTML structure here
            # Example: Find a table and extract rows/cells
            # data_table = soup.find('table', class_='technical-data-table') # Example selector
            # if data_table:
            #    for row in data_table.find_all('tr'):
            #        cells = row.find_all('td') # or 'th'
            #        if len(cells) == 2:
            #             key = cells[0].get_text(strip=True).replace(':', '').strip()
            #             value = cells[1].get_text(strip=True)
            #             if key and value:
            #                 extracted_texts.append(f"{key}: {value}")
            logger.warning(f"HTML cleaning logic for TraceParts is not implemented yet.")
            pass # Placeholder

        # Add logic for other sites if needed
        else:
            logger.warning(f"No specific HTML cleaning logic defined for site: {site_name}. Returning raw text content as fallback.")
            # Fallback: return just the text content of the whole block
            return soup.get_text(separator=' ', strip=True)

        if not extracted_texts:
            logger.warning(f"HTML cleaning for {site_name} resulted in no text extracted.")
            return None # Return None if nothing was extracted

        return "\\n".join(extracted_texts)

    except Exception as e:
        logger.error(f"Error cleaning HTML for {site_name}: {e}", exc_info=True)
        return None # Return None on parsing error

# --- Web Scraping Function (Revised to call cleaner) ---
async def scrape_website_table_html(part_number: str) -> Optional[str]:
    """
    Attempts to scrape the outer HTML of a features table, then cleans it.
    """
    if not part_number:
        logger.debug("Web scraping skipped: No part number provided.")
        return None

    logger.info(f"Attempting web scrape for features table / Part#: '{part_number}'...")

    for site_config in WEBSITE_CONFIGS:
        selector = site_config.get("table_selector")
        site_name = site_config.get("name", "Unknown Site") # Get site name for cleaner
        if not selector:
             logger.warning(f"No table_selector defined for {site_name}. Skipping.")
             continue

        target_url = site_config["base_url_template"].format(part_number=part_number)
        js_code = site_config.get("pre_extraction_js")
        logger.debug(f"Attempting scrape on {site_name} ({target_url}) for table selector '{selector}'")

        # Configure crawler run - Use JsonCssExtractionStrategy to get outerHTML
        extraction_schema = {
            "name": "TableHTML",
            "baseSelector": "html", # Apply to whole document
            "fields": [
                # Try type: "html" to get the inner/outer HTML of the element
                {"name": "html_content", "selector": selector, "type": "html"}
            ]
        }
        run_config = CrawlerRunConfig(
                 cache_mode=CacheMode.BYPASS,
                 js_code=[js_code] if js_code else None,
                 page_timeout=20000,
                 verbose=False, # Set to True for detailed crawl4ai logs
                 extraction_strategy=JsonCssExtractionStrategy(extraction_schema) # Add strategy
            )
        browser_config = BrowserConfig(verbose=False) # Headless default

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Pass the single run_config object
                results = await crawler.arun_many(urls=[target_url], config=run_config)
                result = results[0]

                # Check for success and extracted content from the strategy
                if result.success and result.extracted_content:
                    raw_html = None
                    try:
                        extracted_data_list = json.loads(result.extracted_content)
                        if extracted_data_list and isinstance(extracted_data_list, list) and len(extracted_data_list) > 0:
                            first_item = extracted_data_list[0]
                            if isinstance(first_item, dict) and "html_content" in first_item:
                                raw_html = str(first_item["html_content"]).strip()
                        else:
                            logger.debug(f"Extraction strategy did not find or extract HTML for selector '{selector}' on {site_name}.")

                    except json.JSONDecodeError:
                         logger.warning(f"Failed to parse JSON from crawl4ai extraction result for table HTML on {site_name}: {result.extracted_content[:100]}...")
                    except Exception as parse_error:
                         logger.error(f"Error processing extracted JSON for {site_name}: {parse_error}", exc_info=True)

                    # --- Pass raw HTML to cleaner --- 
                    if raw_html:
                        cleaned_text = clean_scraped_html(raw_html, site_name)
                        if cleaned_text:
                            logger.success(f"Successfully scraped and cleaned features table from {site_name}.")
                            return cleaned_text # Return the cleaned text
                        else:
                             logger.warning(f"HTML was scraped from {site_name}, but cleaning failed or yielded no text.")
                    # else: (already logged failure to extract HTML)

                elif result.error_message:
                     logger.warning(f"Scraping page failed for {site_name} ({target_url}): {result.error_message}")
                else:
                    logger.debug(f"Scraping attempt for {site_name} yielded no extracted content or error message.")

        except asyncio.TimeoutError:
             logger.warning(f"Scraping timed out for {site_name} ({target_url})")
        except Exception as e:
            logger.error(f"Unexpected error during web scraping for {site_name} ({target_url}): {e}", exc_info=True)

    logger.info(f"Web scraping finished for features table. No usable cleaned text found across configured sites.")
    return None


# --- Extraction Chain (Revised Prompt Template - MINOR CHANGE) ---
def create_extraction_chain(retriever, llm):
    """
    Creates a RAG chain that uses both PDF context and potentially cleaned scraped web table data
    to answer an extraction instruction, prioritizing the scraped data.
    """
    if retriever is None or llm is None:
        logger.error("Retriever or LLM is not initialized for extraction chain.")
        return None

    # --- Updated Template to reflect cleaned data ---
    template = """
You are an expert data extractor. Your goal is to extract a specific piece of information based on the Extraction Instructions provided below.
You are given two potential sources of information:
1. Cleaned Scraped Website Data: This is cleaned text extracted from a specific section (likely features/specifications) of a supplier website for the part number. THIS SOURCE IS PREFERRED AND SHOULD BE USED IF THE INFORMATION IS PRESENT AND CLEAR.
2. Document Context: These are text chunks extracted from PDF documents related to the part number. Use this as a fallback if the Cleaned Scraped Website Data is missing, doesn't contain the required information, or is ambiguous.

Part Number Information (if provided by user):
{part_number}

--- PREFERRED SOURCE ---
Cleaned Scraped Website Data:
{scraped_table_html} 
--- END PREFERRED SOURCE ---

--- FALLBACK SOURCE ---
Document Context (from PDFs):
{context}
--- END FALLBACK SOURCE ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: Respond with ONLY a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the extracted result determined by following the Extraction Instructions, prioritizing the 'Cleaned Scraped Website Data' if available and relevant.
- Provide the value as a JSON string. Examples of possible values include "GF, T", "none", "NOT FOUND", "Female", "7.2", "999".
- If the information cannot be found in EITHER the Cleaned Scraped Website Data OR the Document Context based on the instructions, the value should be "NOT FOUND".
- Do NOT include any explanations, reasoning, or any text outside of the single JSON object in your response.

Example Output Format:
{{"{attribute_key}": "extracted_value_from_web_or_pdf"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    # Define the extraction chain using LCEL
    # Takes 'extraction_instructions', 'attribute_key', 'part_number', and 'scraped_table_html' as input
    # Note: 'scraped_table_html' placeholder now holds the CLEANED TEXT
    extraction_chain = (
        RunnableParallel(
            # Retrieve PDF context based on the attribute_key and part_number
            context=RunnablePassthrough() | (lambda x: retriever.invoke(f"Extract information about {x['attribute_key']} for part number {x.get('part_number', 'N/A')}")) | format_docs,
            # Pass through other inputs directly
            extraction_instructions=RunnablePassthrough(),
            attribute_key=RunnablePassthrough(),
            part_number=RunnablePassthrough(),
            scraped_table_html=RunnablePassthrough() # Pass the cleaned text
        )
        # Assign the inputs correctly to the prompt variables
        # Ensure all inputs exist in the dictionary passed to assign
        .assign(
            extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key']['attribute_key'],
            part_number=lambda x: x['part_number'].get('part_number', "Not Provided"),
            scraped_table_html=lambda x: x['scraped_table_html'].get('scraped_table_html', "Not Available") # Get cleaned text safely
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Extraction RAG chain (with web scrape priority) created successfully.")
    return extraction_chain


# --- Simplified run_extraction (Calls the updated chain) ---
@logger.catch(reraise=True)
async def run_extraction(extraction_instructions: str, attribute_key: str, extraction_chain, part_number: str, scraped_table_html: Optional[str]):
    """
    Runs the extraction RAG chain, providing both PDF context (via retriever in chain)
    and potentially pre-scraped web table HTML. The chain's prompt prioritizes the HTML.
    """
    if not attribute_key:
        logger.warning("Received empty attribute key.")
        return '{"error": "No attribute key provided."}'
    if not extraction_chain:
        logger.error("Extraction chain is not available.")
        return '{"error": "Extraction chain is not initialized."}'
    if not extraction_instructions:
        logger.warning(f"Received empty extraction instructions for '{attribute_key}'. LLM might struggle.")
        # Proceed, but the LLM might return "NOT FOUND" or hallucinate without instructions.

    try:
        log_msg = f"Invoking extraction chain for key: '{attribute_key}' with Part Number: '{part_number if part_number else 'None'}'"
        if scraped_table_html:
             log_msg += " (using scraped web HTML)."
        else:
             log_msg += " (no scraped web HTML available, using PDF context)."
        logger.info(log_msg)

        # Prepare input data for the chain
        input_data = {
            "extraction_instructions": extraction_instructions,
            "attribute_key": attribute_key,
            "part_number": part_number if part_number else "Not Provided",
            "scraped_table_html": scraped_table_html if scraped_table_html else "Not Available" # Pass HTML or placeholder
        }

        # Use ainvoke for the async chain
        response = await extraction_chain.ainvoke(input_data)
        logger.info(f"Extraction chain invoked successfully for '{attribute_key}'.")

        # --- Post-processing LLM response (same as before) ---
        cleaned_response = response
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        start_index = cleaned_response.find(think_start_tag)
        end_index = cleaned_response.find(think_end_tag)
        if start_index != -1 and end_index != -1 and end_index > start_index:
             cleaned_response = cleaned_response[end_index + len(think_end_tag):].strip()

        if cleaned_response.strip().startswith("```json"):
            cleaned_response = cleaned_response.strip()[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        # Attempt to validate/load JSON before returning
        try:
             # Basic validation: does it start with { and end with }?
             if not (cleaned_response.startswith('{') and cleaned_response.endswith('}')):
                  raise json.JSONDecodeError("Does not look like a JSON object.", cleaned_response, 0)
             # Try full parse
             json.loads(cleaned_response)
             return cleaned_response
        except json.JSONDecodeError as json_err:
             logger.error(f"LLM for '{attribute_key}' returned invalid JSON after cleaning: {json_err}. Response: '{cleaned_response}' Raw: '{response}'")
             # Return error JSON including the raw response for debugging in the UI
             err_payload = {"error": f"LLM returned invalid JSON: {json_err}", "raw_llm_output": response}
             return json.dumps(err_payload) # Ensure the error itself is valid JSON

    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
             logger.error(f"Rate limit hit during LLM extraction for '{attribute_key}': {e}", exc_info=False)
             return json.dumps({"error": f"API Rate Limit Hit for {attribute_key}"}) # Valid JSON error
        else:
             logger.error(f"Error invoking LLM extraction chain for '{attribute_key}': {e}", exc_info=True)
             return json.dumps({"error": f"An error occurred during LLM extraction: {str(e)}"}) # Valid JSON error


# --- Remove old scraping function and placeholders ---
# async def scrape_website_for_attribute(attribute_key: str, part_number: str) -> Optional[str]: ...
# TARGET_URLS = [ ... ]
# ATTRIBUTE_SELECTORS = { ... }