# pdf_processor.py
import os
import re
from typing import List, BinaryIO
from loguru import logger # Using Loguru for nice logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document

import config # Import configuration

def clean_text(text: str) -> str:
    """Applies basic cleaning to extracted text."""
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    text = text.replace('-\n', '') # Handle hyphenation (simple case)
    text = re.sub(r'\n\s*\n', '\n', text) # Remove excessive newlines
    # Add more specific cleaning rules if needed
    return text

def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:
    """Process uploaded PDFs without chunking, keeping full document context."""
    all_docs = []
    saved_file_paths = []
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
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
                
                # Combine all pages into a single document
                combined_content = "\n\n".join(doc.page_content for doc in documents)
                cleaned_content = clean_text(combined_content)
                
                if cleaned_content:
                    # Create a single document with the combined content
                    combined_doc = Document(
                        page_content=cleaned_content,
                        metadata={
                            'source': file_basename,
                            'pages': len(documents)
                        }
                    )
                    all_docs.append(combined_doc)
                    logger.success(f"Successfully processed {file_basename}")
                else:
                    logger.warning(f"No processable content found in {file_basename} after cleaning.")
                    
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
    
    logger.info(f"Total documents processed: {len(all_docs)}")
    return all_docs