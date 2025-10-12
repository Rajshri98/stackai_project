# LLM-based reranking module for advanced relevance scoring

import json
import re
import requests
from typing import List, Dict, Tuple, Optional
from settings import MISTRAL_API_KEY
from constants import LLM_RERANK_TIMEOUT

class LLMReranker:
    """
    Advanced reranking using LLM to judge relevance between query and chunks
    """
    
    def __init__(self):
        self.api_key = MISTRAL_API_KEY
        self.model = "mistral-small-latest"
        
    def rerank_chunks(self, query: str, chunks: List[Dict], chunk_metadata: List[Dict]) -> List[Dict]:
        """
        Rerank chunks using LLM-based relevance scoring
        
        Args:
            query: User query
            chunks: List of search result dictionaries with scores
            chunk_metadata: List of chunk metadata with text content
            
        Returns:
            List of chunks with added LLM scores and reasoning
        """
        if not self.api_key or not chunks:
            # Fallback: return original chunks with default LLM scores
            for chunk in chunks:
                chunk["llm_score"] = 5.0  # Neutral score
                chunk["llm_reasoning"] = "LLM reranking unavailable"
            return chunks
        
        reranked_chunks = []
        
        for chunk in chunks:
            # Get the chunk text from metadata
            chunk_idx = chunk["idx"]
            chunk_text = chunk_metadata[chunk_idx]["text"]
            
            # Get LLM relevance score
            llm_score, reasoning = self._score_relevance(query, chunk_text)
            
            # Add LLM scores to chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk["llm_score"] = llm_score
            enhanced_chunk["llm_reasoning"] = reasoning
            
            reranked_chunks.append(enhanced_chunk)
        
        # Sort by LLM score (descending)
        reranked_chunks.sort(key=lambda x: x["llm_score"], reverse=True)
        
        return reranked_chunks
    
    def _score_relevance(self, query: str, chunk_text: str) -> Tuple[float, str]:
        """
        Score the relevance of a chunk to a query using LLM
        
        Args:
            query: User query
            chunk_text: Text content of the chunk
            
        Returns:
            Tuple of (score, reasoning)
        """
        prompt = self._build_rerank_prompt(query, chunk_text)
        
        try:
            response = self._call_mistral(prompt)
            score, reasoning = self._parse_rerank_response(response)
            return score, reasoning
            
        except Exception as e:
            print(f"LLM reranking error: {e}")
            return 5.0, f"Error in LLM scoring: {str(e)}"
    
    def _build_rerank_prompt(self, query: str, chunk_text: str) -> str:
        """Build the reranking prompt for LLM"""
        # Truncate chunk text if too long (to fit in context)
        max_chunk_length = 800
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
        
        prompt = f"""Rate the relevance of this text chunk to the user's query on a scale of 1-10.

Query: "{query}"

Text Chunk:
{chunk_text}

Consider:
- Direct relevance to the question asked
- Completeness of information provided
- Context accuracy and usefulness
- How well it answers the specific query

Respond in this exact format:
Score: [number 1-10]
Reasoning: [brief explanation]

Example:
Score: 8
Reasoning: Directly answers the question about experience with specific technologies mentioned in the query.

Your response:"""

        return prompt
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral API for reranking"""
        if not self.api_key:
            raise Exception("Mistral API key not available")
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Low temperature for consistent scoring
            "max_tokens": 150
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=LLM_RERANK_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    
    def _parse_rerank_response(self, response: str) -> Tuple[float, str]:
        """
        Parse LLM response to extract score and reasoning
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (score, reasoning)
        """
        try:
            # Look for "Score: X" pattern
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                # Clamp score to 1-10 range
                score = max(1.0, min(10.0, score))
            else:
                # Fallback: look for any number
                number_match = re.search(r'(\d+(?:\.\d+)?)', response)
                if number_match:
                    score = max(1.0, min(10.0, float(number_match.group(1))))
                else:
                    score = 5.0  # Default neutral score
            
            # Extract reasoning
            reasoning_match = re.search(r'Reasoning:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Limit reasoning length
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
            else:
                reasoning = "No reasoning provided"
            
            return score, reasoning
            
        except Exception as e:
            print(f"Error parsing rerank response: {e}")
            return 5.0, f"Parse error: {str(e)}"
    
    def combine_scores(self, chunks: List[Dict], llm_weight: float = 0.7, hybrid_weight: float = 0.3) -> List[Dict]:
        """
        Combine LLM scores with original hybrid scores
        
        Args:
            chunks: List of chunks with both hybrid and LLM scores
            llm_weight: Weight for LLM scores
            hybrid_weight: Weight for hybrid scores
            
        Returns:
            List of chunks with combined final scores
        """
        for chunk in chunks:
            # Normalize LLM score to 0-1 range (from 1-10)
            normalized_llm = (chunk.get("llm_score", 5.0) - 1.0) / 9.0
            
            # Original hybrid score is already 0-1
            hybrid_score = chunk.get("score", 0.0)
            
            # Combine scores
            final_score = (llm_weight * normalized_llm) + (hybrid_weight * hybrid_score)
            
            chunk["final_score"] = final_score
            chunk["normalized_llm_score"] = normalized_llm
        
        # Sort by final combined score
        chunks.sort(key=lambda x: x["final_score"], reverse=True)
        
        return chunks