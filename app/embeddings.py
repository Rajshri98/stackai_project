# Custom TF-IDF and Word Embeddings implementation

import numpy as np
import re
from typing import List, Union, Dict
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CustomTFIDF:
    """Custom TF-IDF implementation for keyword-based search"""
    
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
        self.fitted = False
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF on documents"""
        self.vectorizer.fit(documents)
        self.fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return self.vectorizer.transform(documents).toarray()
    
    def transform_single(self, text: str) -> np.ndarray:
        """Transform single text to TF-IDF vector"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        return self.vectorizer.transform([text]).toarray()[0]
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (vocabulary)"""
        return self.vectorizer.get_feature_names_out().tolist()

class CustomWordEmbeddings:
    """Custom word embeddings using TF-IDF weighted word vectors"""
    
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.tfidf = CustomTFIDF()
        self.fitted = False
    
    def _create_word_vectors(self, documents: List[str]):
        """Create simple word vectors based on TF-IDF weights"""
        # Fit TF-IDF
        self.tfidf.fit(documents)
        tfidf_matrix = self.tfidf.transform(documents)
        feature_names = self.tfidf.get_feature_names()
        
        # Create word vectors based on TF-IDF weights
        for i, doc in enumerate(documents):
            words = re.findall(r'\b\w+\b', doc.lower())
            for word in words:
                if word in feature_names:
                    word_idx = feature_names.index(word)
                    weight = tfidf_matrix[i, word_idx]
                    
                    if word not in self.word_vectors:
                        # Initialize with random vector
                        self.word_vectors[word] = np.random.normal(0, 0.1, self.embedding_dim)
                    
                    # Update vector with TF-IDF weight
                    self.word_vectors[word] += weight * np.random.normal(0, 0.01, self.embedding_dim)
    
    def fit(self, documents: List[str]):
        """Fit word embeddings on documents"""
        self._create_word_vectors(documents)
        self.fitted = True
    
    def _get_document_embedding(self, text: str) -> np.ndarray:
        """Get document embedding by averaging word vectors"""
        words = re.findall(r'\b\w+\b', text.lower())
        vectors = []
        
        for word in words:
            if word in self.word_vectors:
                vectors.append(self.word_vectors[word])
        
        if not vectors:
            return np.zeros(self.embedding_dim)
        
        return np.mean(vectors, axis=0)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.fitted:
            raise ValueError("Must fit before encode")
        
        embeddings = []
        for text in texts:
            embedding = self._get_document_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.fitted:
            raise ValueError("Must fit before encode")
        return self._get_document_embedding(text)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dim

class EmbeddingGenerator:
    """Handles text-to-vector embedding generation using custom implementations"""
    
    def __init__(self):
        """Initialize the embedding models"""
        self.tfidf = CustomTFIDF()
        self.word_embeddings = CustomWordEmbeddings()
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """Fit both TF-IDF and word embeddings on texts"""
        self.tfidf.fit(texts)
        self.word_embeddings.fit(texts)
        self.fitted = True
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.fitted:
            raise ValueError("Must fit before encode")
        return self.word_embeddings.encode(texts)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.fitted:
            raise ValueError("Must fit before encode")
        return self.word_embeddings.encode_single(text)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.word_embeddings.get_embedding_dimension()
    
    def get_tfidf_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """Get TF-IDF similarity scores between query and documents"""
        if not self.fitted:
            raise ValueError("Must fit before getting similarity")
        
        query_vector = self.tfidf.transform_single(query)
        doc_vectors = self.tfidf.transform(documents)
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_vector], doc_vectors)[0]
        return similarities
