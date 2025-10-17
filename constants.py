# Configuration for the retrieval + generation pipeline

# Chunking
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200  

# Embedding model
EMBEDDING_MODEL = "custom-tfidf-word-embeddings"
EMBEDDING_DIMENSION = 300  

# Storage paths
DATA_DIR = "data"
CHUNKS_JSON = "data/chunks_metadata.json"
FAISS_INDEX = "data/vector_index.npy"  

# Hybrid retrieval weighting
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# Multi-stage retrieval parameters
RETRIEVAL_CANDIDATES = 25  
RERANK_TOP_K = 8  
FINAL_TOP_K = 2  

# Scoring thresholds
MIN_THRESHOLD = 0.001  
EVIDENCE_THRESHOLD = 0.001  

# LLM reranking
LLM_RERANK_ENABLED = True
LLM_RERANK_WEIGHT = 0.7  
HYBRID_WEIGHT = 0.3 
LLM_RERANK_TIMEOUT = 10  

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75