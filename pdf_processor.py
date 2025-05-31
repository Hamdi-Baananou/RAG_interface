# pdf_processor.py
import os
import re
from typing import List, BinaryIO
from loguru import logger # Using Loguru for nice logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config # Import configuration

def clean_text(text: str) -> str:
    """Applies basic cleaning to extracted text."""
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    text = text.replace('-\n', '') # Handle hyphenation (simple case)
    text = re.sub(r'\n\s*\n', '\n', text) # Remove excessive newlines
    # Add more specific cleaning rules if needed
    return text

def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:
    """Process uploaded PDFs with chunking, maintaining document context."""
    all_docs = []
    saved_file_paths = []
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize text splitter with config values
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    
    try:
        for uploaded_file in uploaded_files:
            file_basename = uploaded_file.name
            file_path = os.path.join(temp_dir, file_basename)
            saved_file_paths.append(file_path)
            
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                logger.info(f"Loading PDF: {file_basename}")
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()  # List of Docs, one per page
                
                if not documents:
                    logger.warning(f"No pages extracted from {file_basename}")
                    continue
                
                # Process each page and maintain metadata
                for doc in documents:
                    # Clean the page content
                    cleaned_content = clean_text(doc.page_content)
                    
                    if cleaned_content:
                        # Split the cleaned content into chunks
                        chunks = text_splitter.split_text(cleaned_content)
                        
                        # Create Document objects for each chunk with metadata
                        for i, chunk in enumerate(chunks):
                            chunk_doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': file_basename,
                                    'page': doc.metadata.get('page', 'N/A'),
                                    'chunk': i + 1,
                                    'total_chunks': len(chunks)
                                }
                            )
                            all_docs.append(chunk_doc)
                        
                        logger.success(f"Successfully processed page {doc.metadata.get('page', 'N/A')} from {file_basename}")
                    else:
                        logger.warning(f"No processable content found in page {doc.metadata.get('page', 'N/A')} of {file_basename} after cleaning.")
                    
            except Exception as e:
                logger.error(f"Error processing {file_basename}: {e}", exc_info=True)
                
    finally:
        # Clean up temporary files
        for path in saved_file_paths:
            try:
                os.remove(path)
                logger.debug(f"Removed temporary file: {path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {path}: {e}")
    
    if not all_docs:
        logger.error("No text could be extracted from any provided PDF files.")
    
    logger.info(f"Total document chunks processed: {len(all_docs)}")
    return all_docs