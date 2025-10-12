import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from constants import CHUNKS_JSON, FAISS_INDEX, DATA_DIR, EMBEDDING_DIMENSION
from sklearn.metrics.pairwise import cosine_similarity

def save_chunks_and_vectors(chunks: List[Dict], embeddings: np.ndarray):
    """
    Save chunk metadata to JSON and vector embeddings to numpy array
    
    Args:
        chunks: List of chunk dictionaries with metadata
        embeddings: numpy array of embeddings with shape (len(chunks), embedding_dim)
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save chunk metadata to JSON
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save embeddings as numpy array (replacing FAISS)
    np.save(FAISS_INDEX.replace('.faiss', '.npy'), embeddings.astype('float32'))
    
    print(f"Saved {len(chunks)} chunks and {embeddings.shape[0]} vectors")

def load_chunks_and_vectors() -> Tuple[List[Dict], Optional[np.ndarray]]:
    """
    Load chunk metadata and vector embeddings
    
    Returns:
        Tuple of (chunks_list, embeddings_array) or ([], None) if files don't exist
    """
    # Check if files exist
    if not os.path.exists(CHUNKS_JSON):
        return [], None
    
    # Load chunk metadata
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Load embeddings if they exist
    embeddings = None
    embeddings_file = FAISS_INDEX.replace('.faiss', '.npy')
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    
    return chunks, embeddings

def load_chunks():
    """Legacy function for backward compatibility"""
    chunks, _ = load_chunks_and_vectors()
    return [chunk["text"] for chunk in chunks] if chunks else []

def save_chunks(chunks):
    """Legacy function for backward compatibility - saves only text chunks"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)