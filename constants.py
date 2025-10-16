# Configuration for the retrieval + generation pipeline

# Chunking
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Embedding model
EMBEDDING_MODEL = "custom-tfidf-word-embeddings"
EMBEDDING_DIMENSION = 300  # Custom word embeddings dimension

# Storage paths
DATA_DIR = "data"
CHUNKS_JSON = "data/chunks_metadata.json"
FAISS_INDEX = "data/vector_index.npy"  # Changed to .npy for numpy arrays

# Hybrid retrieval weighting
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# Multi-stage retrieval parameters
RETRIEVAL_CANDIDATES = 25  # Stage 1: Retrieve many candidates
RERANK_TOP_K = 8  # Stage 2: How many to send to LLM reranking
FINAL_TOP_K = 2  # Stage 3: Final results to return

# Scoring thresholds
MIN_THRESHOLD = 0.001  # Even lower for stage 1 (more inclusive)
EVIDENCE_THRESHOLD = 0.001  # Even lower final threshold for better retrieval

# LLM reranking
LLM_RERANK_ENABLED = True
LLM_RERANK_WEIGHT = 0.7  # Weight for LLM scores vs hybrid scores
HYBRID_WEIGHT = 0.3  # Weight for original hybrid scores
LLM_RERANK_TIMEOUT = 10  # Seconds to wait for LLM response

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75