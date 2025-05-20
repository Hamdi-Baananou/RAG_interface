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
import aiohttp
from datetime import datetime, timedelta

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, ContentTypeFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# Add TraceParts API configuration
TRACEPARTS_API_CONFIG = {
    "base_url": "https://api.traceparts.com/v3",
    "catalog": "TE_CONNECTIVITY",
    "api_key": None,  # Will be set from environment
    "token": None,
    "token_expiry": None
}

# Initialize TraceParts API key from config
TRACEPARTS_API_CONFIG["api_key"] = config.TRACEPARTS_API_KEY
if not TRACEPARTS_API_CONFIG["api_key"]:
    logger.warning("TRACEPARTS_API_KEY not found in environment variables. API access will be disabled.")

# Add TraceParts API functions
async def get_traceparts_token() -> Optional[str]:
    """Get or refresh the TraceParts API token."""
    if (TRACEPARTS_API_CONFIG["token"] and TRACEPARTS_API_CONFIG["token_expiry"] and 
        datetime.now() < TRACEPARTS_API_CONFIG["token_expiry"]):
        return TRACEPARTS_API_CONFIG["token"]
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{TRACEPARTS_API_CONFIG['base_url']}/auth/token",
                json={"apiKey": TRACEPARTS_API_CONFIG["api_key"]}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    TRACEPARTS_API_CONFIG["token"] = data.get("token")
                    # Set token expiry to 23 hours (giving 1 hour buffer)
                    TRACEPARTS_API_CONFIG["token_expiry"] = datetime.now() + timedelta(hours=23)
                    return TRACEPARTS_API_CONFIG["token"]
                else:
                    logger.error(f"Failed to get TraceParts token: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error getting TraceParts token: {e}")
        return None

async def get_traceparts_product_data(part_number: str) -> Optional[Dict]:
    """Get product data from TraceParts API."""
    token = await get_traceparts_token()
    if not token:
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {token}"}
            params = {
                "catalog": TRACEPARTS_API_CONFIG["catalog"],
                "partNumber": part_number
            }
            
            async with session.get(
                f"{TRACEPARTS_API_CONFIG['base_url']}/product",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 401:
                    # Token might be expired, try refreshing
                    TRACEPARTS_API_CONFIG["token"] = None
                    token = await get_traceparts_token()
                    if token:
                        # Retry the request with new token
                        headers = {"Authorization": f"Bearer {token}"}
                        async with session.get(
                            f"{TRACEPARTS_API_CONFIG['base_url']}/product",
                            headers=headers,
                            params=params
                        ) as retry_response:
                            if retry_response.status == 200:
                                data = await retry_response.json()
                                return data
                logger.error(f"Failed to get TraceParts product data: {response.status}")
                return None
    except Exception as e:
        logger.error(f"Error getting TraceParts product data: {e}")
        return None

def format_traceparts_data(product_data: Dict) -> str:
    """Format TraceParts API response into key-value pairs."""
    if not product_data:
        return None
    
    extracted_texts = []
    
    # Extract basic product information
    if "product" in product_data:
        product = product_data["product"]
        
        # Add basic properties
        for key, value in product.items():
            if isinstance(value, (str, int, float, bool)):
                extracted_texts.append(f"{key}: {value}")
        
        # Add specifications if available
        if "specifications" in product:
            for spec in product["specifications"]:
                if "name" in spec and "value" in spec:
                    extracted_texts.append(f"{spec['name']}: {spec['value']}")
        
        # Add features if available
        if "features" in product:
            for feature in product["features"]:
                if isinstance(feature, dict) and "name" in feature:
                    extracted_texts.append(f"Feature: {feature['name']}")
    
    return "\\n".join(extracted_texts) if extracted_texts else None

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

# --- LLM-Free Web Scraping Configuration (Revised for Full Page) ---

# Configure websites to scrape, in order of preference.
WEBSITE_CONFIGS = [
    {
        "name": "TraceParts",
        "base_url_template": "https://www.traceparts.com/en/search?CatalogPath=&KeepFilters=true&Keywords={part_number}&SearchAction=Keywords",
        "pre_extraction_js": """
            async function scrapeTraceParts() {
                try {
                    console.log('Starting TraceParts scraping process...');
                    
                    // Wait for search results to load
                    console.log('Waiting for search results...');
                    await new Promise(r => setTimeout(r, 5000));
                    
                    // Look for search results container
                    const searchResults = document.getElementById('search-results-items');
                    console.log('Search results container found:', !!searchResults);
                    
                    if (!searchResults) {
                        console.log('Search results container not found, waiting longer...');
                        await new Promise(r => setTimeout(r, 5000));
                        const searchResults = document.getElementById('search-results-items');
                        if (!searchResults) {
                            console.log('Still no search results found after additional wait');
                            return false;
                        }
                    }
                    
                    // Find the card containing the exact part number
                    const cards = searchResults.querySelectorAll('.card');
                    console.log('Found ' + cards.length + ' result cards');
                    
                    let foundMatch = false;
                    for (const card of cards) {
                        const partNumberSpan = card.querySelector('.partnumber');
                        if (partNumberSpan) {
                            const foundPartNumber = partNumberSpan.textContent.trim();
                            console.log('Checking part number: ' + foundPartNumber);
                            
                            if (foundPartNumber === '{part_number}') {
                                console.log('Found exact part match, looking for link...');
                                // Find and click the link
                                const link = card.querySelector('a.row');
                                console.log('Link found:', !!link);
                                
                                if (link) {
                                    console.log('Clicking link to product page...');
                                    link.click();
                                    
                                    // Wait for navigation to complete by checking URL
                                    console.log('Waiting for navigation to complete...');
                                    let navigationComplete = false;
                                    await new Promise(r => {
                                        const i = setInterval(() => {
                                            if (location.href.includes('{part_number}')) {
                                                clearInterval(i);
                                                navigationComplete = true;
                                                r();
                                            }
                                        }, 500);
                                    });
                                    
                                    if (!navigationComplete) {
                                        console.log('Navigation did not complete successfully');
                                        return false;
                                    }
                                    
                                    // Additional wait for page load
                                    console.log('Waiting for page load...');
                                    await new Promise(r => setTimeout(r, 3000));
                                    
                                    // Scroll to make specs visible and trigger AJAX load
                                    console.log('Scrolling to make specs visible...');
                                    const specsElement = document.querySelector('.tp-product-specifications');
                                    if (specsElement) {
                                        specsElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                        await new Promise(r => setTimeout(r, 2000));
                                    } else {
                                        // Fallback scroll if element not found
                                        window.scrollBy(0, 800);
                                        await new Promise(r => setTimeout(r, 2000));
                                    }
                                    
                                    // Try to find and expand any technical data sections
                                    const expandButtons = document.querySelectorAll('.technical-data-expander, .expander-button, [aria-expanded]');
                                    console.log('Found ' + expandButtons.length + ' expand buttons');
                                    
                                    for (const button of expandButtons) {
                                        console.log('Button text:', button.textContent);
                                        console.log('Button aria-expanded:', button.getAttribute('aria-expanded'));
                                        if (button.getAttribute('aria-expanded') === 'false') {
                                            console.log('Expanding section...');
                                            button.click();
                                            await new Promise(r => setTimeout(r, 1000));
                                        }
                                    }
                                    
                                    // Final wait to ensure all content is loaded
                                    await new Promise(r => setTimeout(r, 2000));
                                    foundMatch = true;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (!foundMatch) {
                        console.log('No exact part number match found in results');
                    }
                    
                    return foundMatch;
                    
                } catch (error) {
                    console.error('Error during TraceParts scraping:', error);
                    console.error('Error stack:', error.stack);
                }
                return false;
            }
            return scrapeTraceParts();
        """
    },
    {
        "name": "Mouser",
        "base_url_template": "https://www.mouser.com/Search/Refine?Keyword={part_number}",
        "pre_extraction_js": """
            async function scrapeMouser() {
                try {
                    // Wait for search results to load
                    await new Promise(r => setTimeout(r, 2000));
                    
                    // Click on the first result if found
                    const firstResult = document.querySelector('.product-list-item a');
                    if (firstResult) {
                        console.log('Clicking first search result...');
                        firstResult.click();
                        await new Promise(r => setTimeout(r, 2000));
                    }
                } catch (error) {
                    console.error('Error during Mouser scraping:', error);
                }
            }
            scrapeMouser();
        """
    },
    {
        "name": "TE Connectivity",
        "base_url_template": "https://www.te.com/en/product-{part_number}.html",
        "pre_extraction_js": """
            async function scrapeTE() {
                try {
                    const expandButtonSelector = '#pdp-features-expander-btn';
                    const expandButton = document.querySelector(expandButtonSelector);
                    
                    if (expandButton && expandButton.getAttribute('aria-selected') === 'false') {
                        console.log('Features expand button indicates collapsed state, clicking...');
                        expandButton.click();
                        await new Promise(r => setTimeout(r, 1500));
                    }
                } catch (error) {
                    console.error('Error during TE Connectivity scraping:', error);
                }
            }
            scrapeTE();
        """
    }
]

# --- HTML Cleaning Function ---
def clean_scraped_html(html_content: str, site_name: str) -> Optional[str]:
    """
    Parses scraped HTML using BeautifulSoup and extracts key-value pairs
    from known structures for different supplier websites.
    """
    if not html_content:
        return None

    logger.debug(f"Cleaning HTML content from {site_name}...")
    soup = BeautifulSoup(html_content, 'html.parser')
    extracted_texts = []

    try:
        if site_name == "TraceParts":
            # Try multiple selectors for specifications
            specs_selectors = [
                '.tp-product-specifications',
                '.technical-data-table',
                '.product-details-table',
                '.specifications-table',
                '#product-specifications',
                '.product-specifications'
            ]
            
            for selector in specs_selectors:
                elements = soup.select(selector)
                for element in elements:
                    # Try to find key-value pairs in different formats
                    # Format 1: List items with title and value classes
                    for li in element.select('li'):
                        title = li.select_one('.title, .spec-title, .spec-name')
                        value = li.select_one('.value, .spec-value, .spec-data')
                        if title and value:
                            key = title.get_text(strip=True).replace(':', '').strip()
                            val = value.get_text(strip=True)
                            if key and val:
                                extracted_texts.append(f"{key}: {val}")
                    
                    # Format 2: Table rows
                    for row in element.select('tr'):
                        cells = row.select('td, th')
                        if len(cells) >= 2:
                            key = cells[0].get_text(strip=True).replace(':', '').strip()
                            value = cells[1].get_text(strip=True)
                            if key and value:
                                extracted_texts.append(f"{key}: {value}")
                    
                    # Format 3: Div pairs
                    for div in element.select('div'):
                        if div.find_previous_sibling('div'):
                            key = div.find_previous_sibling('div').get_text(strip=True).replace(':', '').strip()
                            value = div.get_text(strip=True)
                            if key and value:
                                extracted_texts.append(f"{key}: {value}")

            # If still no specs found, try to extract any text that looks like a specification
            if not extracted_texts:
                for text in soup.stripped_strings:
                    if ':' in text:
                        key, value = text.split(':', 1)
                        if key.strip() and value.strip():
                            extracted_texts.append(f"{key.strip()}: {value.strip()}")

            logger.info(f"Extracted {len(extracted_texts)} features from TraceParts.")

        elif site_name == "Mouser":
            # Try multiple selectors for Mouser
            specs_selectors = [
                '.product-details-table',
                '.specifications-table',
                '.product-specifications',
                '.product-attributes',
                '.product-features'
            ]
            
            for selector in specs_selectors:
                elements = soup.select(selector)
                for element in elements:
                    # Try different formats
                    # Format 1: Table rows
                    for row in element.select('tr'):
                        cells = row.select('td, th')
                        if len(cells) >= 2:
                            key = cells[0].get_text(strip=True).replace(':', '').strip()
                            value = cells[1].get_text(strip=True)
                            if key and value:
                                extracted_texts.append(f"{key}: {value}")
                    
                    # Format 2: List items
                    for li in element.select('li'):
                        text = li.get_text(strip=True)
                        if ':' in text:
                            key, value = text.split(':', 1)
                            if key.strip() and value.strip():
                                extracted_texts.append(f"{key.strip()}: {value.strip()}")

            logger.info(f"Extracted {len(extracted_texts)} features from Mouser.")

        elif site_name == "TE Connectivity":
            # Try multiple selectors for TE
            specs_selectors = [
                '.product-features',
                '.product-specifications',
                '.technical-specifications',
                '#pdp-features-tabpanel',
                '.specifications-panel'
            ]
            
            for selector in specs_selectors:
                elements = soup.select(selector)
                for element in elements:
                    # Try different formats
                    # Format 1: Feature items
                    for item in element.select('li.product-feature, .feature-item'):
                        title = item.select_one('.feature-title, .title')
                        value = item.select_one('.feature-value, .value')
                        if title and value:
                            key = title.get_text(strip=True).replace(':', '').strip()
                            val = value.get_text(strip=True)
                            if key and val:
                                extracted_texts.append(f"{key}: {val}")
                    
                    # Format 2: Any text with colon
                    for text in element.stripped_strings:
                        if ':' in text:
                            key, value = text.split(':', 1)
                            if key.strip() and value.strip():
                                extracted_texts.append(f"{key.strip()}: {value.strip()}")

            logger.info(f"Extracted {len(extracted_texts)} features from TE Connectivity.")

        # If no specific site logic worked, try generic extraction
        if not extracted_texts:
            logger.warning(f"No specific extraction logic worked for {site_name}, trying generic extraction...")
            # Look for any text that looks like a key-value pair
            for text in soup.stripped_strings:
                if ':' in text:
                    key, value = text.split(':', 1)
                    if key.strip() and value.strip():
                        extracted_texts.append(f"{key.strip()}: {value.strip()}")

        if not extracted_texts:
            logger.warning(f"HTML cleaning for {site_name} resulted in no text extracted.")
            return None

        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in extracted_texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        return "\\n".join(unique_texts)

    except Exception as e:
        logger.error(f"Error cleaning HTML for {site_name}: {e}", exc_info=True)
        return None

# --- Web Scraping Function (Using Crawl4AI's BestFirstCrawling) ---
async def scrape_website_table_html(part_number: str) -> Optional[Dict[str, str]]:
    """
    Attempts to scrape product data using Crawl4AI's BestFirstCrawling strategy.
    Returns a dictionary containing both the cleaned text and the source website.
    """
    if not part_number:
        logger.debug("Web scraping skipped: No part number provided.")
        return None

    logger.info(f"Attempting web scrape for part number: '{part_number}'...")

    # Configure websites to try, in order of preference
    websites = [
        {
            "name": "TraceParts",
            "url_template": "https://www.traceparts.com/en/search?CatalogPath=&KeepFilters=true&Keywords={part_number}&SearchAction=Keywords",
            "keywords": ["specification", "technical", "product", "details", "features"],
            "patterns": ["*product*", "*specification*", "*technical*"],
            "wait_time": 5000,  # 5 seconds wait for dynamic content
            "scroll_script": """
                async function scrollAndWait() {
                    // Wait for initial load
                    await new Promise(r => setTimeout(r, 2000));
                    
                    // Scroll to bottom
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    
                    // Scroll back to top
                    window.scrollTo(0, 0);
                    await new Promise(r => setTimeout(r, 1000));
                    
                    // Look for and click on product link
                    const links = Array.from(document.querySelectorAll('a'));
                    const productLink = links.find(link => 
                        link.href && 
                        link.href.includes('{part_number}') && 
                        !link.href.includes('search')
                    );
                    
                    if (productLink) {
                        productLink.click();
                        await new Promise(r => setTimeout(r, 3000));
                        
                        // Scroll to specifications
                        const specsElement = document.querySelector('.tp-product-specifications, .technical-data-table');
                        if (specsElement) {
                            specsElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            await new Promise(r => setTimeout(r, 2000));
                        }
                    }
                }
                return scrollAndWait();
            """
        },
        {
            "name": "Mouser",
            "url_template": "https://www.mouser.com/Search/Refine?Keyword={part_number}",
            "keywords": ["specification", "technical", "product", "details", "features"],
            "patterns": ["*product*", "*specification*", "*technical*"],
            "wait_time": 3000,  # 3 seconds wait for dynamic content
            "scroll_script": """
                async function scrollAndWait() {
                    // Wait for initial load
                    await new Promise(r => setTimeout(r, 2000));
                    
                    // Click first product result
                    const firstResult = document.querySelector('.product-list-item a');
                    if (firstResult) {
                        firstResult.click();
                        await new Promise(r => setTimeout(r, 2000));
                        
                        // Scroll to specifications
                        const specsElement = document.querySelector('.product-details-table, .specifications-table');
                        if (specsElement) {
                            specsElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            await new Promise(r => setTimeout(r, 1000));
                        }
                    }
                }
                return scrollAndWait();
            """
        }
    ]

    for site in websites:
        site_name = site["name"]
        target_url = site["url_template"].format(part_number=part_number)
        
        try:
            logger.info(f"Attempting to crawl {site_name}...")
            
            # Create a keyword relevance scorer
            keyword_scorer = KeywordRelevanceScorer(
                keywords=site["keywords"],
                weight=0.7
            )

            # Create a filter chain
            filter_chain = FilterChain([
                URLPatternFilter(patterns=site["patterns"]),
                ContentTypeFilter(allowed_types=["text/html"])
            ])

            # Configure the crawler with enhanced browser settings
            browser_config = BrowserConfig(
                verbose=True,
                headless=True,
                wait_for_network_idle=True,
                wait_time=site["wait_time"],
                scroll_script=site["scroll_script"].format(part_number=part_number)
            )

            # Configure the crawler
            config = CrawlerRunConfig(
                deep_crawl_strategy=BestFirstCrawlingStrategy(
                    max_depth=2,
                    include_external=False,
                    filter_chain=filter_chain,
                    url_scorer=keyword_scorer,
                    max_pages=3  # Reduced to focus on main product page
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                stream=False,
                verbose=True
            )

            # Execute the crawl
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun(target_url, config=config)
                
                if results and isinstance(results, list):
                    # Get the best result
                    best_result = max(results, key=lambda r: r.metadata.get('score', 0))
                    
                    # Try multiple ways to get content
                    content = None
                    if hasattr(best_result, '_results') and best_result._results:
                        first_result = best_result._results[0]
                        # Try different content attributes
                        for attr in ['html', 'cleaned_html', 'extracted_content', 'content']:
                            if hasattr(first_result, attr):
                                content = getattr(first_result, attr)
                                if content:
                                    break
                    
                    if content:
                        logger.info(f"Successfully loaded content from {site_name}. Length: {len(content)} characters.")
                        logger.debug(f"Content preview: {content[:500]}...")
                        
                        # Clean the HTML content
                        cleaned_text = clean_scraped_html(content, site_name)
                        
                        if cleaned_text:
                            logger.success(f"Successfully scraped and cleaned data from {site_name}.")
                            return {
                                "text": cleaned_text,
                                "source": site_name,
                                "url": best_result.url if hasattr(best_result, 'url') else target_url
                            }
                        else:
                            logger.warning(f"No cleaned text could be extracted from {site_name}.")
                    else:
                        logger.warning(f"Could not find page content in result for {site_name}.")
                else:
                    logger.warning(f"No results found for {site_name}.")

        except Exception as e:
            logger.error(f"Unexpected error during web scraping for {site_name} ({target_url}): {e}", exc_info=True)

    logger.info(f"Web scraping finished. No usable data found across configured sites.")
    return None


# --- PDF Extraction Chain (Using Retriever and Detailed Instructions) ---
def create_pdf_extraction_chain(retriever, llm):
    """
    Creates a RAG chain that uses ONLY PDF context (via retriever)
    and detailed instructions to answer an extraction task.
    """
    if retriever is None or llm is None:
        logger.error("Retriever or LLM is not initialized for PDF extraction chain.")
        return None

    # Template using only PDF context and detailed instructions passed at runtime
    template = """
You are an expert data extractor. Your goal is to extract a specific piece of information based on the Extraction Instructions provided below, using ONLY the Document Context from PDFs.

Part Number Information (if provided by user):
{part_number}

--- Document Context (from PDFs) ---
{context}
--- End Document Context ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: Respond with ONLY a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the extracted result determined by following the Extraction Instructions using the Document Context provided above.
- Provide the value as a JSON string. Examples: "GF, T", "none", "NOT FOUND", "Female", "7.2", "999".
- Do NOT include any explanations, reasoning, or any text outside of the single JSON object in your response.

Example Output Format:
{{"{attribute_key}": "extracted_value_from_pdf"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    # Chain uses retriever to get PDF context
    pdf_chain = (
        RunnableParallel(
            context=RunnablePassthrough() | (lambda x: retriever.invoke(f"Extract information about {x['attribute_key']} for part number {x.get('part_number', 'N/A')}")) | format_docs,
            extraction_instructions=RunnablePassthrough(),
            attribute_key=RunnablePassthrough(),
            part_number=RunnablePassthrough()
        )
        .assign(
            extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key']['attribute_key'],
            part_number=lambda x: x['part_number'].get('part_number', "Not Provided")
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("PDF Extraction RAG chain created successfully.")
    return pdf_chain

# --- Web Data Extraction Chain (Using Cleaned Web Text and Simple Prompt) ---
def create_web_extraction_chain(llm):
    """
    Creates a simpler chain that uses ONLY cleaned website data
    and a direct instruction to extract an attribute strictly.
    """
    if llm is None:
        logger.error("LLM is not initialized for Web extraction chain.")
        return None

    # Simplified template allowing reasoning based on web data and instructions
    template = """
You are an expert data extractor. Your goal is to answer a specific piece of information by applying the logic described in the 'Extraction Instructions' to the 'Cleaned Scraped Website Data' provided below. Use ONLY the provided website data as your context.

--- Cleaned Scraped Website Data ---
{cleaned_web_data}
--- End Cleaned Scraped Website Data ---

Extraction Instructions:
{extraction_instructions}

---
IMPORTANT: Follow the Extraction Instructions carefully using the website data.
Respond with ONLY a single, valid JSON object containing exactly one key-value pair.
- The key for the JSON object MUST be the string: "{attribute_key}"
- The value MUST be the result obtained by applying the Extraction Instructions to the Cleaned Scraped Website Data.
- Provide the value as a JSON string.
- If the information cannot be determined from the Cleaned Scraped Website Data based on the instructions, the value MUST be "NOT FOUND".
- Do NOT include any explanations or reasoning outside the JSON object.

Example Output Format:
{{"{attribute_key}": "extracted_value_based_on_instructions"}}

Output:
"""
    prompt = PromptTemplate.from_template(template)

    # Chain structure similar to PDF chain to handle inputs
    web_chain = (
        RunnableParallel(
            cleaned_web_data=RunnablePassthrough(),
            extraction_instructions=RunnablePassthrough(),
            attribute_key=RunnablePassthrough()
        )
        .assign(
            cleaned_web_data=lambda x: x['cleaned_web_data']['cleaned_web_data'], # Nested dict access
            extraction_instructions=lambda x: x['extraction_instructions']['extraction_instructions'],
            attribute_key=lambda x: x['attribute_key']['attribute_key']
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("Web Data Extraction chain created successfully (accepts instructions).")
    return web_chain


# --- Helper function to invoke chain and process response (KEEP THIS) ---
async def _invoke_chain_and_process(chain, input_data, attribute_key):
    """Helper to invoke chain, handle errors, and clean response."""
    response = await chain.ainvoke(input_data)
    log_msg = f"Chain invoked successfully for '{attribute_key}'."
    # Add response length to log for debugging potential truncation/verboseness
    if response:
         log_msg += f" Response length: {len(response)}"
    logger.info(log_msg)

    if response is None:
         logger.error(f"Chain invocation returned None for '{attribute_key}'")
         return json.dumps({"error": f"Chain invocation returned None for {attribute_key}"})

    # --- Enhanced Cleaning --- 
    cleaned_response = response
    
    # 1. Remove <think> tags (already handled)
    think_start_tag = "<think>"
    think_end_tag = "</think>"
    start_index_think = cleaned_response.find(think_start_tag)
    end_index_think = cleaned_response.find(think_end_tag)
    if start_index_think != -1 and end_index_think != -1 and end_index_think > start_index_think:
         cleaned_response = cleaned_response[end_index_think + len(think_end_tag):].strip()

    # 2. Remove ```json ... ``` markdown (already handled)
    if cleaned_response.strip().startswith("```json"):
        cleaned_response = cleaned_response.strip()[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

    # 3. Find the first '{' and the last '}' to isolate the JSON object
    try:
        first_brace = cleaned_response.find('{')
        last_brace = cleaned_response.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = cleaned_response[first_brace : last_brace + 1]
            # Attempt to parse the isolated part
            json.loads(potential_json) # Test if it's valid JSON
            cleaned_response = potential_json # If valid, use this isolated part
            logger.debug(f"Isolated potential JSON for '{attribute_key}': {cleaned_response}")
        else:
             logger.warning(f"Could not find clear JSON braces {{...}} in response for '{attribute_key}'. Using original cleaned response.")
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse isolated JSON for '{attribute_key}'. Using original cleaned response. Raw: {cleaned_response}")
        # If parsing the isolated part fails, fall back to the previously cleaned response
        pass 
    except Exception as e:
         logger.error(f"Unexpected error during JSON isolation for '{attribute_key}': {e}")
         # Fallback
         pass
    # --- End Enhanced Cleaning ---

    return cleaned_response # Validation happens in the caller (app.py now)


# --- REMOVE Unified Chain and Old run_extraction ---
# def create_extraction_chain(retriever, llm): ...
# @logger.catch(reraise=True)
# async def run_extraction(...): ...