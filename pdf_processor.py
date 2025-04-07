# pdf_processor.py
import os
import re
from typing import List, BinaryIO
from loguru import logger # Using Loguru for nice logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import config # Import configuration

# Initialize text splitter globally or within function if params change often
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len, # Standard way to measure length
    add_start_index=True, # Helpful for context tracking
)

def clean_text(text: str) -> str:
    """Applies basic cleaning to extracted text."""
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    text = text.replace('-\n', '') # Handle hyphenation (simple case)
    text = re.sub(r'\n\s*\n', '\n', text) # Remove excessive newlines
    # Add more specific cleaning rules if needed
    return text

def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:
    """
    Loads, cleans, and splits text from uploaded PDF files.
    Requires saving uploaded files temporarily as PyMuPDFLoader needs file paths.
    Args:
        uploaded_files: List of file-like objects from Streamlit's file_uploader.
        temp_dir: Directory to temporarily store PDF files.

    Returns:
        List of split Document objects.
    """
    all_split_docs = []
    os.makedirs(temp_dir, exist_ok=True) # Ensure temp dir exists

    if not uploaded_files:
        logger.warning("No PDF files provided for processing.")
        return []

    saved_file_paths = []
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer()) # Save the uploaded file content
            saved_file_paths.append(file_path)
            logger.info(f"Temporarily saved uploaded file: {uploaded_file.name}")

            file_basename = os.path.basename(file_path)
            try:
                logger.info(f"Loading PDF: {file_basename}")
                loader = PyMuPDFLoader(file_path)
                documents = loader.load() # List of Docs, one per page

                if not documents:
                    logger.warning(f"No pages extracted from {file_basename}")
                    continue

                processed_pages = []
                for i, doc in enumerate(documents):
                    cleaned_content = clean_text(doc.page_content)
                    if cleaned_content: # Only process pages with actual content after cleaning
                        doc.page_content = cleaned_content
                        # Ensure necessary metadata is present
                        doc.metadata['source'] = file_basename
                        doc.metadata['page'] = doc.metadata.get('page_number', i) # PyMuPDF uses 0-index
                        processed_pages.append(doc)
                    else:
                        logger.debug(f"Page {i} of {file_basename} has no content after cleaning.")


                if not processed_pages:
                    logger.warning(f"No processable content found in {file_basename} after cleaning.")
                    continue

                logger.info(f"Splitting text from {len(processed_pages)} pages of {file_basename}...")
                split_docs = text_splitter.split_documents(processed_pages)

                if not split_docs:
                    logger.warning(f"No text chunks generated after splitting {file_basename}")
                    continue

                logger.success(f"Generated {len(split_docs)} chunks from {file_basename}")
                all_split_docs.extend(split_docs)

            except Exception as e:
                logger.error(f"Error processing {file_basename}: {e}", exc_info=True)
                # Continue with the next file

    finally:
        # Clean up temporary files
        for path in saved_file_paths:
            try:
                os.remove(path)
                logger.debug(f"Removed temporary file: {path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {path}: {e}")
        # Optionally remove temp dir if empty, but might be reused
        # if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        #     os.rmdir(temp_dir)

    if not all_split_docs:
        logger.error("No text could be extracted and split from any provided PDF files.")
        # Consider raising an exception or returning a specific status

    logger.info(f"Total chunks generated from all files: {len(all_split_docs)}")
    return all_split_docs