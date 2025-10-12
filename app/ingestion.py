# Extracts text from PDFs, cleans and chunks it, then writes
# chunk metadata to JSON and a NumPy vector index to disk.

# Extract text from pdfs uploaded
import fitz
import os
from typing import List
from fastapi import UploadFile
from app.utils_text import clean_text, chunk_text
from app.embeddings import EmbeddingGenerator
from app.store import save_chunks_and_vectors
from constants import CHUNK_SIZE, CHUNK_OVERLAP

async def process_pdfs(files: List[UploadFile]):
    """
    Process uploaded PDF files using PyMuPDF: extract text, build chunks with
    metadata, generate embeddings, and persist both metadata (JSON) and the
    NumPy vector index to disk.

    Args:
        files: Uploaded PDF files

    Returns:
        Dict with processing status and counts
    """
    # Initialize embedding generator
    embedding_gen = EmbeddingGenerator()
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    all_chunks = []

    for file in files:
        try:
            # Extract text from PDF
            file_content = await file.read()
            pdf = fitz.open(stream=file_content, filetype="pdf")
            text = ""

            # Concatenate text from all pages
            for page in pdf:
                text += page.get_text("text")
            pdf.close()
            
            # Skip files that yielded no text
            if not text.strip():
                print(f"Warning: No text extracted from {file.filename}")
                continue
                
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            return {"status": "error", "message": f"Failed to process {file.filename}: {str(e)}"}

        # Clean and chunk text with metadata
        cleaned = clean_text(text)
        chunks = chunk_text(
            text=cleaned, 
            filename=file.filename,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"status": "error", "message": "No text chunks created from uploaded files"}

    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    
    # Fit the embedding generator on the documents
    embedding_gen.fit(chunk_texts)
    embeddings = embedding_gen.encode(chunk_texts)
    
    # Save chunks metadata and embeddings
    save_chunks_and_vectors(all_chunks, embeddings)

    return {
        "status": "ok", 
        "chunks_created": len(all_chunks),
        "files_processed": len(files),
        "embedding_dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
    }