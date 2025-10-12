# Text cleaning and chunking with character-based overlap

import re
from typing import List, Dict
from constants import CHUNK_SIZE, CHUNK_OVERLAP

# Clean text
def clean_text(text: str) -> str:
    """Clean and normalize text by removing extra whitespace"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Split text into overlapping chunks with metadata
def chunk_text(text: str, filename: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into overlapping chunks with metadata
    
    Args:
        text: Input text to chunk
        filename: Source filename for metadata
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    chunk_id = 0
    
    # Handle edge case where text is shorter than chunk_size
    if len(text) <= chunk_size:
        if text.strip():
            chunks.append({
                "id": f"{filename}_{chunk_id}",
                "text": text.strip(),
                "filename": filename,
                "chunk_size": len(text.strip()),
                "start_pos": 0,
                "end_pos": len(text.strip())
            })
        return chunks
    
    # Create overlapping chunks
    step_size = chunk_size - overlap
    for i in range(0, len(text), step_size):
        chunk_text = text[i:i + chunk_size]
        
        # Skip empty chunks
        if chunk_text.strip():
            chunks.append({
                "id": f"{filename}_{chunk_id}",
                "text": chunk_text.strip(),
                "filename": filename,
                "chunk_size": len(chunk_text.strip()),
                "start_pos": i,
                "end_pos": i + len(chunk_text)
            })
            chunk_id += 1
            
        # Break if we've reached the end
        if i + chunk_size >= len(text):
            break
    
    return chunks