# pdf_processor.py
import os
import re
import base64
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, BinaryIO, Optional, Dict, Any
from loguru import logger
from PIL import Image
import fitz  # PyMuPDF
from mistralai import Mistral
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

# Global thread pool for PDF processing
pdf_thread_pool = ThreadPoolExecutor(max_workers=2)  # Adjust based on your needs

def encode_pil_image(pil_image: Image.Image, format: str = "PNG") -> tuple[str, str]:
    """Encode PIL Image to Base64 string."""
    buffered = io.BytesIO()
    # Ensure image is in RGB mode
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    save_format = format.upper()
    if save_format not in ["PNG", "JPEG"]:
        logger.warning(f"Unsupported format '{format}', defaulting to PNG.")
        save_format = "PNG"

    pil_image.save(buffered, format=save_format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8'), save_format.lower()

async def process_single_pdf(file_path: str, file_basename: str, client: Mistral, model_name: str, 
                           text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Process a single PDF file and return its documents."""
    all_docs = []
    total_pages_processed = 0
    pdf_document = None
    
    try:
        logger.info(f"Processing PDF: {file_basename}")
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(file_path)
        total_pages = len(pdf_document)
        logger.info(f"Successfully opened PDF with {total_pages} pages")
        
        # Define the prompt for Mistral Vision
        markdown_prompt = """
You are an expert document analysis assistant. Extract ALL text content from the image and format it as clean, well-structured GitHub Flavored Markdown.

Follow these formatting instructions:
1. Use appropriate Markdown heading levels based on visual hierarchy
2. Format tables using GitHub Flavored Markdown table syntax
3. Format key-value pairs using bold for keys: `**Key:** Value`
4. Represent checkboxes as `[x]` or `[ ]`
5. Preserve bulleted/numbered lists using standard Markdown syntax
6. Maintain paragraph structure and line breaks
7. Extract text labels from diagrams/images
8. Ensure all visible text is captured accurately

Output only the generated Markdown content.
"""
        
        for page_num in range(total_pages):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing page {page_num + 1}/{total_pages} of {file_basename}")
            logger.info(f"{'='*50}\n")
            
            try:
                # Get the page
                page = pdf_document[page_num]
                
                # Convert page to image with higher resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Encode image to base64
                base64_image, image_format = encode_pil_image(img)
                
                # Prepare message for Mistral Vision
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": markdown_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        ]
                    }
                ]
                
                # Call Mistral Vision API
                logger.info("Sending page to Mistral Vision API...")
                chat_response = client.chat.complete(
                    model=model_name,
                    messages=messages
                )
                
                # Get extracted text
                page_content = chat_response.choices[0].message.content
                
                if page_content:
                    # Log the extracted content
                    logger.info("\nExtracted Content:")
                    logger.info("-" * 40)
                    logger.info(page_content)
                    logger.info("-" * 40)
                    
                    # Split the content into chunks
                    chunks = text_splitter.split_text(page_content)
                    logger.info(f"\nSplit content into {len(chunks)} chunks")
                    
                    # Create Document objects for each chunk
                    for j, chunk in enumerate(chunks):
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata={
                                'source': file_basename,
                                'page': page_num + 1,
                                'chunk': j + 1,
                                'total_chunks': len(chunks)
                            }
                        )
                        all_docs.append(chunk_doc)
                    
                    logger.success(f"Successfully processed page {page_num + 1} from {file_basename}")
                    total_pages_processed += 1
                else:
                    logger.warning(f"No content extracted from page {page_num + 1} of {file_basename}")
                    
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1} with Mistral Vision: {e}")
                
    except Exception as e:
        logger.error(f"Error processing {file_basename}: {e}", exc_info=True)
    finally:
        # Close the PDF document if it was opened
        if pdf_document is not None:
            try:
                pdf_document.close()
                logger.debug(f"Closed PDF document: {file_basename}")
            except Exception as e:
                logger.warning(f"Error closing PDF document {file_basename}: {e}")
    
    if not all_docs:
        logger.error(f"No text could be extracted from {file_basename}")
    else:
        logger.info(f"\nProcessing Summary for {file_basename}:")
        logger.info(f"Total pages processed: {total_pages_processed}")
        logger.info(f"Total chunks created: {len(all_docs)}")
    
    return all_docs

async def process_uploaded_pdfs(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> List[Document]:
    """Process uploaded PDFs using Mistral Vision for better text extraction."""
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
    
    # Initialize Mistral client
    try:
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        model_name = config.VISION_MODEL_NAME
        logger.info(f"Initialized Mistral Vision client with model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client: {e}")
        return []
    
    try:
        # Save all files first
        for uploaded_file in uploaded_files:
            file_basename = uploaded_file.name
            file_path = os.path.join(temp_dir, file_basename)
            saved_file_paths.append(file_path)
            
            # Save uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        # Process PDFs in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(saved_file_paths), 4)) as executor:
            # Create tasks for each PDF
            loop = asyncio.get_event_loop()
            tasks = []
            for file_path in saved_file_paths:
                file_basename = os.path.basename(file_path)
                # Create a task that runs in the thread pool
                task = loop.run_in_executor(
                    executor,
                    lambda p, b: asyncio.run(process_single_pdf(p, b, client, model_name, text_splitter)),
                    file_path,
                    file_basename
                )
                tasks.append(task)
            
            # Wait for all PDFs to be processed
            results = await asyncio.gather(*tasks)
            
            # Combine all results
            for docs in results:
                if docs:  # Check if docs is not None
                    all_docs.extend(docs)
            
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
    else:
        logger.info("\nFinal Processing Summary:")
        logger.info(f"Total documents processed: {len(saved_file_paths)}")
        logger.info(f"Total chunks created: {len(all_docs)}")
    
    return all_docs

def process_pdfs_in_background(uploaded_files: List[BinaryIO], temp_dir: str = "temp_pdf") -> asyncio.Task:
    """Start PDF processing in the background and return a task that can be awaited later."""
    return asyncio.create_task(process_uploaded_pdfs(uploaded_files, temp_dir))