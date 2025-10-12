# Query pipeline: retrieval, LLM reranking, and answer generation

import re
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from app.store import load_chunks_and_vectors
from app.embeddings import EmbeddingGenerator
from settings import MISTRAL_API_KEY
from app.reranker import LLMReranker
from constants import (BM25_WEIGHT, VECTOR_WEIGHT, MIN_THRESHOLD, EVIDENCE_THRESHOLD, 
                      RETRIEVAL_CANDIDATES, RERANK_TOP_K, FINAL_TOP_K, 
                      LLM_RERANK_ENABLED, LLM_RERANK_WEIGHT, HYBRID_WEIGHT)
from sklearn.metrics.pairwise import cosine_similarity

# Lazy singleton for the embedding generator
_embedding_generator = None

# Lightweight stopword list for overlap checks
_STOPWORDS = {
    "the","a","an","and","or","but","if","in","on","at","to","of","for","by",
    "with","about","as","is","are","was","were","be","been","being","this","that",
    "those","these","from","it","its","into","how","what","when","where","who","why",
    "hi","hello","hey","hows","life","doing","you","your","yours","me","my","we","our"
}

def get_embedding_generator():
    """Get or create embedding generator instance"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
class IntentClassifier:
    """Lightweight semantic intent classifier used only to guide formatting/routing."""
    def __init__(self):
        self.embedding_gen = None
        self.labels: List[str] = [
            "greeting",            # hello, hi, hey
            "clarification",       # how does this work, can you help
            "list",                # list, enumerate, show all
            "comparison",          # compare, vs, difference
            "timeline",            # when, timeline, chronology
            "skills",              # skills, qualifications, expertise
            "experience",          # experience, jobs, positions
            "education",           # education, degree, university
            "contact",             # contact, email, phone
            "kb"                    # default knowledge/document question
        ]
        self.examples: Dict[str, List[str]] = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "clarification": ["i don't understand", "how does this work", "can you help", "what should i do"],
            "list": ["list", "show me all", "enumerate", "name all"],
            "comparison": ["compare", "difference", "vs", "versus", "contrast"],
            "timeline": ["when", "timeline", "chronology", "history", "sequence"],
            "skills": ["skills", "qualifications", "expertise", "abilities", "competencies"],
            "experience": ["experience", "work", "jobs", "career", "positions"],
            "education": ["education", "degree", "university", "college", "school"],
            "contact": ["contact", "email", "phone", "address", "reach"],
            "kb": ["what is", "summarize", "explain", "describe", "tell me about", "who", "where", "why", "how"]
        }
        self._embeddings: Optional[Dict[str, np.ndarray]] = None

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        # Use a dedicated embedding generator to avoid interfering with retrieval embeddings
        self.embedding_gen = EmbeddingGenerator()
        corpus: List[str] = []
        index: List[Tuple[str, int]] = []
        for label, exs in self.examples.items():
            for ex in exs:
                index.append((label, len(corpus)))
                corpus.append(ex)
        # Fit on examples only (kept small and generic)
        self.embedding_gen.fit(corpus)
        enc = self.embedding_gen.encode(corpus)
        buckets: Dict[str, List[np.ndarray]] = {k: [] for k in self.labels}
        for (label, i), vec in zip(index, enc):
            buckets[label].append(vec)
        self._embeddings = {k: np.vstack(v) if v else np.zeros((1, self.embedding_gen.get_embedding_dimension())) for k, v in buckets.items()}

    def classify(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return "kb"
        self._ensure_embeddings()
        qv = self.embedding_gen.encode_single(q)
        best_label, best_score = "kb", -1.0
        for label, mat in self._embeddings.items():
            sims = cosine_similarity([qv], mat)[0]
            score = float(np.max(sims)) if sims.size else -1.0
            if score > best_score:
                best_label, best_score = label, score
        # Apply a minimal confidence to reduce random misclassifications
        return best_label if best_score >= 0.15 else "kb"

intent_classifier = IntentClassifier()

def get_intent_embeddings():
    """
    Generate embeddings for all intent examples.
    Returns a dictionary with intent categories and their embeddings.
    """
    embedding_gen = get_embedding_generator()
    
    # Fit the embedding generator on all examples
    all_examples = []
    for examples in INTENT_EXAMPLES.values():
        all_examples.extend(examples)
    
    embedding_gen.fit(all_examples)
    
    # Generate embeddings for each intent category
    intent_embeddings = {}
    for intent, examples in INTENT_EXAMPLES.items():
        embeddings = embedding_gen.encode(examples)
        intent_embeddings[intent] = embeddings
    
    return intent_embeddings

def classify_by_similarity(query: str, intent_embeddings: dict) -> str:
    """
    Classify query intent using semantic similarity matching.
    Returns the most similar intent category.
    """
    if not query.strip():
        return 'other'

    embedding_gen = get_embedding_generator()
    query_embedding = embedding_gen.encode_single(query)

    best_intent = 'other'
    best_similarity = -1

    # Compare query to each intent category
    for intent, examples_embeddings in intent_embeddings.items():
        # Calculate similarity to all examples in this intent
        similarities = cosine_similarity([query_embedding], examples_embeddings)[0]

        # Use max similarity as the score for this intent
        max_similarity = np.max(similarities)

        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_intent = intent

    # Only return the intent if similarity is above threshold
    if best_similarity > 0.1:  # Much lower threshold for better matching
        return best_intent
    else:
        return 'other'

def determine_response_type(query: str) -> str:
    """
    Determine the appropriate response format based on query type.
    """
    query_lower = query.lower()
    
    # List-type questions
    if any(word in query_lower for word in ['list', 'what are', 'name all', 'show me all', 'enumerate']):
        return "a bulleted list with clear, concise items"
    
    # Comparison questions
    if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'contrast']):
        return "a structured comparison with clear sections"
    
    # Timeline/chronological questions
    if any(word in query_lower for word in ['when', 'timeline', 'chronology', 'history', 'sequence']):
        return "a chronological timeline or ordered list"
    
    # Skills/qualifications questions
    if any(word in query_lower for word in ['skills', 'qualifications', 'expertise', 'abilities', 'competencies']):
        return "a categorized list of skills and qualifications"
    
    # Experience/work history questions
    if any(word in query_lower for word in ['experience', 'work', 'jobs', 'career', 'positions']):
        return "a structured list of work experience with details"
    
    # Education questions
    if any(word in query_lower for word in ['education', 'degree', 'university', 'college', 'school']):
        return "a structured list of educational background"
    
    # Contact information
    if any(word in query_lower for word in ['contact', 'email', 'phone', 'address', 'reach']):
        return "a clear list of contact information"
    
    # Default to paragraph format
    return "a clear, well-structured paragraph"

def _extract_title_like_phrases(text: str, max_items: int = 6) -> List[str]:
    """No specialized extraction. Kept only for compatibility; returns empty list."""
    return []

def generate_structured_fallback(query: str, context: str, response_type: str) -> str:
    """Generic fallback: summarize the stitched context into bullets or a paragraph."""
    text = (context or "").strip()
    if not text:
        return "I don't have enough relevant data to answer confidently."

    # Split into sentences (lightweight, generic)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Pick the first few informative sentences
    selected = sentences[:5] if sentences else [text[:200]]

    if "list" in response_type or "bulleted" in response_type or "structured" in response_type:
        return "\n".join(f"• {s}" for s in selected)

    # Default to a compact paragraph
    return " ".join(selected)[:1200]


def _looks_like_header(text: str) -> bool:
    """Heuristic: detect contact-like headers; avoids dumping boilerplate as answers."""
    if not text:
        return False
    header_keywords = [
        "email", "phone", "linkedin", "website", "summary", "objective",
        "address", "new york", "gmail.com", "@", "|"
    ]
    lines = [l.strip().lower() for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    # If the first block contains many contact-like tokens, treat as header
    joined = " ".join(lines[:5])
    hits = sum(1 for k in header_keywords if k in joined)
    return hits >= 2

def needs_search(query: str) -> bool:
    """Always search documents (placeholder kept for compatibility)."""
    return True

# Normalize and Transform
def normalize(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t]

def transform_query(query: str) -> str:
    q = query.lower().strip()
    # Light phrase normalization
    q = re.sub(r"\bwhat\s+is\b", "definition of", q)
    q = re.sub(r"\bhow\s+to\b", "steps to", q)

    # Synonym expansion for common colloquialisms that hurt retrieval
    # Document-agnostic synonym expansion. Keep generic to work across domains.
    synonyms = {
        "listings": ["items", "entries", "records", "mentions"],
        "overview": ["summary", "abstract", "high level"],
        "diff": ["difference", "compare", "contrast"],
        "pros": ["advantages", "benefits"],
        "cons": ["disadvantages", "limitations"],
        "contact": ["email", "phone"],
    }
    for term, exps in synonyms.items():
        if term in q:
            q += " " + " ".join(exps)

    return q

# Custom TF-IDF keyword search
def tfidf_search(query: str, chunks: List[Dict], embedding_gen: EmbeddingGenerator, top_k: int = 20) -> List[Dict]:
    """
    Perform TF-IDF keyword search on text chunks
    
    Args:
        query: Query string
        chunks: List of chunk dictionaries with metadata
        embedding_gen: EmbeddingGenerator instance
        top_k: Number of top results to return
        
    Returns:
        List of results with TF-IDF scores
    """
    if not chunks or not query.strip():
        return []
    
    # Get chunk texts
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    # Get TF-IDF similarity scores
    similarities = embedding_gen.get_tfidf_similarity(query, chunk_texts)
    
    # Create results with metadata
    results = []
    for idx, score in enumerate(similarities):
        results.append({
            "idx": idx,
            "id": chunks[idx]["id"],
            "bm25_score": float(score),  # Keep same field name for compatibility
            "filename": chunks[idx]["filename"]
        })
    
    # Sort by TF-IDF score and return top_k
    return sorted(results, key=lambda x: x["bm25_score"], reverse=True)[:top_k]

# Custom vector similarity search using numpy
def vector_search(query_embedding: np.ndarray, embeddings: np.ndarray, chunks: List[Dict], top_k: int = 20) -> List[Dict]:
    """
    Perform semantic vector search using cosine similarity
    
    Args:
        query_embedding: Query embedding vector
        embeddings: Stored embeddings array
        chunks: List of chunk dictionaries with metadata
        top_k: Number of top results to return
        
    Returns:
        List of results with vector similarity scores
    """
    if embeddings is None or len(chunks) == 0:
        return []
    
    # Calculate cosine similarity between query and all embeddings
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top_k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "idx": int(idx),
            "id": chunks[idx]["id"],
            "vector_score": float(similarities[idx]),
            "filename": chunks[idx]["filename"]
        })
    
    return results

# Multi-stage retrieval: Stage 1 - Retrieve many candidates
def hybrid_search_stage1(query: str, chunks: List[Dict], embeddings: np.ndarray, top_k: int = RETRIEVAL_CANDIDATES) -> List[Dict]:
    """
    Stage 1: Retrieve many candidates using hybrid TF-IDF + Vector search
    
    Args:
        query: User query string
        chunks: List of chunk dictionaries
        embeddings: Stored embeddings array
        top_k: Number of candidates to retrieve (default: 25)
        
    Returns:
        List of candidate results with hybrid scores
    """
    if not chunks:
        return []
    
    embedding_gen = get_embedding_generator()
    # Ensure TF-IDF/embeddings are fitted on current chunk corpus (singleton is fresh on process start)
    if not getattr(embedding_gen, 'fitted', False):
        try:
            embedding_gen.fit([c["text"] for c in chunks])
        except Exception:
            # If fitting fails, proceed; downstream will handle empty similarities gracefully
            pass
    
    # Get TF-IDF results (more candidates)
    tfidf_results = tfidf_search(query, chunks, embedding_gen, top_k=top_k*2)
    
    # Get vector results (more candidates)
    vector_results = []
    if embeddings is not None:
        query_embedding = embedding_gen.encode_single(query)
        vector_results = vector_search(query_embedding, embeddings, chunks, top_k=top_k*2)
    
    # Combine results by chunk index
    combined = {}
    
    # Add TF-IDF results
    for result in tfidf_results:
        idx = result["idx"]
        combined[idx] = {
            "idx": idx,
            "id": result["id"],
            "filename": result["filename"],
            "bm25_score": result["bm25_score"],  # Keep same field name
            "vector_score": 0.0
        }
    
    # Add vector results
    for result in vector_results:
        idx = result["idx"]
        if idx in combined:
            combined[idx]["vector_score"] = result["vector_score"]
        else:
            combined[idx] = {
                "idx": idx,
                "id": result["id"],
                "filename": result["filename"],
                "bm25_score": 0.0,
                "vector_score": result["vector_score"]
            }
    
    # Calculate combined scores
    for idx in combined:
        result = combined[idx]
        result["score"] = (
            BM25_WEIGHT * result["bm25_score"] + 
            VECTOR_WEIGHT * result["vector_score"]
        )
        result["semantic_score"] = result["vector_score"]  # For backward compatibility
        result["keyword_score"] = result["bm25_score"]     # For backward compatibility
    
    # Sort by combined score and return top candidates
    candidates = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    
    # Apply minimal filtering for stage 1 (be inclusive)
    filtered_candidates = [c for c in candidates if c["score"] >= MIN_THRESHOLD]
    
    return filtered_candidates[:top_k]

#Evidence threshold + citations
def gather_context(chunks: List[str], results: List[Dict], threshold=0.05, top_k=4):
    if not results or results[0]["score"] < threshold:
        return None, []

    context = []
    cites = []

    for r in results[:top_k]:
        # Collect contextual text with chunk IDs
        context.append(f"[CHUNK {r['idx']}] {chunks[r['idx']]}")
        # Preserve detailed scoring for transparency
        cites.append({
            "chunk": r["idx"],
            "semantic_score": round(r.get("semantic_score", 0), 3),
            "keyword_score": round(r.get("keyword_score", 0), 3),
            "combined_score": round(r.get("score", 0), 3)
        })

    return "\n\n".join(context), cites

# Generation: Call Mistral LLM
def call_mistral(prompt: str) -> str:
    api_key = MISTRAL_API_KEY
    if not api_key:
        return "Mistral API key missing."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            return "API_RATE_LIMIT_ERROR"
        return f"Error communicating with Mistral API: {e}"
    except (KeyError, IndexError):
        return "Unexpected response from Mistral API."


# Post-processing: merge + re-rank

def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a, b = set(a_tokens), set(b_tokens)
    return len(a & b) / (len(a | b) or 1)

def _fuse_by_rrf(results: List[Dict], rrf_k: int = 60) -> List[Dict]:
    # Reciprocal Rank Fusion of semantic vs keyword rankings
    by_sem = sorted(results, key=lambda x: x["semantic_score"], reverse=True)
    by_kw  = sorted(results, key=lambda x: x["keyword_score"],  reverse=True)
    ranks_sem = {r["idx"]: i+1 for i, r in enumerate(by_sem)}
    ranks_kw  = {r["idx"]: i+1 for i, r in enumerate(by_kw)}

    fused = []
    for r in results:
        i = r["idx"]
        rrf = (1 / (rrf_k + ranks_sem.get(i, 1))) + (1 / (rrf_k + ranks_kw.get(i, 1)))
        fused.append({**r, "rrf": round(rrf, 5)})
    return sorted(fused, key=lambda x: (x["rrf"], x["score"]), reverse=True)

def _mmr_select(query_tokens: List[str], candidates: List[Dict], chunks: List[str],
                lambda_balance: float = 0.7, top_k: int = 4) -> List[Dict]:
    # Maximum Marginal Relevance (diverse yet relevant results)
    selected, selected_idxs = [], set()
    cand_tokens = {c["idx"]: normalize(chunks[c["idx"]]) for c in candidates}

    while candidates and len(selected) < top_k:
        best, best_val = None, -1
        for c in candidates:
            if c["idx"] in selected_idxs:
                continue
            relevance = c["score"]
            if selected:
                max_sim = max(_jaccard(cand_tokens[c["idx"]], cand_tokens[s["idx"]]) for s in selected)
            else:
                max_sim = 0
            mmr = lambda_balance * relevance - (1 - lambda_balance) * max_sim
            if mmr > best_val:
                best, best_val = c, mmr
        selected.append(best)
        selected_idxs.add(best["idx"])
        candidates = [x for x in candidates if x["idx"] not in selected_idxs]
    return selected

def _stitch_adjacent(chunks: List[str], picked: List[Dict], window: int = 1) -> (str, List[Dict]):
    # Merge ±1 neighboring chunks 
    indices = set()
    for p in picked:
        i = p["idx"]
        for j in range(max(0, i - window), min(len(chunks), i + window + 1)):
            indices.add(j)

    stitched = "\n\n".join(f"[CHUNK {i}] {chunks[i]}" for i in sorted(indices))

    # Preserve scoring fields 
    citations = []
    for p in picked:
        citations.append({
            "chunk": p["idx"],
            "semantic_score": round(p.get("semantic_score", 0), 3),
            "keyword_score": round(p.get("keyword_score", 0), 3),
            "combined_score": round(p.get("score", 0), 3),
            "rrf": round(p.get("rrf", 0), 3)
        })

    return stitched, citations


def multistage_retrieval(query: str, chunks: List[Dict], embeddings: np.ndarray) -> List[Dict]:
    """
    Multi-stage retrieval with LLM reranking
    
    Stage 1: Retrieve many candidates (25)
    Stage 2: LLM rerank top candidates (8)  
    Stage 3: Final selection (2-3)
    
    Args:
        query: User query string
        chunks: List of chunk dictionaries
        embeddings: Stored embeddings array
        
    Returns:
        List of final reranked results
    """
    if not chunks:
        return []
    
    # Stage 1: Retrieve many candidates using hybrid search
    print(f"Stage 1: Retrieving {RETRIEVAL_CANDIDATES} candidates...")
    candidates = hybrid_search_stage1(query, chunks, embeddings, top_k=RETRIEVAL_CANDIDATES)
    
    if not candidates:
        return []
    
    print(f"Stage 1 complete: {len(candidates)} candidates retrieved")
    
    # Stage 2: LLM reranking (if enabled)
    if LLM_RERANK_ENABLED and len(candidates) > 1:
        print(f"Stage 2: LLM reranking top {min(RERANK_TOP_K, len(candidates))} candidates...")
        
        # Take top candidates for reranking
        rerank_candidates = candidates[:RERANK_TOP_K]
        
        try:
            # Initialize reranker
            reranker = LLMReranker()
            
            # Rerank with LLM
            reranked = reranker.rerank_chunks(query, rerank_candidates, chunks)
            
            # Combine LLM + hybrid scores
            final_candidates = reranker.combine_scores(
                reranked, 
                llm_weight=LLM_RERANK_WEIGHT, 
                hybrid_weight=HYBRID_WEIGHT
            )
            
            print("Stage 2 complete: LLM reranking successful")
            
        except Exception as e:
            print(f"Stage 2 failed: {e}. Falling back to hybrid scores.")
            # Fallback: use original candidates if LLM reranking fails
            final_candidates = candidates
            for candidate in final_candidates:
                candidate["final_score"] = candidate.get("score", 0.0)
                candidate["llm_score"] = 5.0  # Neutral score
                candidate["llm_reasoning"] = f"LLM reranking failed: {str(e)}"
    else:
        print("Stage 2 skipped: LLM reranking disabled or insufficient candidates")
        # Use hybrid scores as final scores
        final_candidates = candidates
        for candidate in final_candidates:
            candidate["final_score"] = candidate.get("score", 0.0)
            candidate["llm_score"] = 5.0  # Neutral score
            candidate["llm_reasoning"] = "LLM reranking disabled"
    
    # Stage 3: Final selection with evidence threshold
    print("Stage 3: Final selection...")
    
    # Filter by evidence threshold
    qualified_candidates = [
        c for c in final_candidates 
        if c.get("final_score", 0.0) >= EVIDENCE_THRESHOLD
    ]
    
    if not qualified_candidates:
        return []
    
    # Return top final candidates
    final_results = qualified_candidates[:FINAL_TOP_K]
    print(f"Stage 3 complete: {len(final_results)} final results selected")
    
    return final_results
def postprocess_multistage_results(chunks: List[Dict], results: List[Dict]) -> tuple[Optional[str], List[Dict]]:
    """
    Post-process multi-stage retrieval results for context generation
    
    Args:
        chunks: List of chunk dictionaries with metadata
        results: Multi-stage retrieval results with LLM scores
        
    Returns:
        Tuple of (context_string, citations) or (None, []) if insufficient evidence
    """
    if not results:
        return None, []
    
    # Generate context and enhanced citations
    stitched, cites = _stitch_adjacent_with_llm_metadata(chunks, results, window=0)
    
    if not stitched.strip():
        return None, []
    
    return stitched, cites

def _stitch_adjacent_with_llm_metadata(chunks: List[Dict], picked: List[Dict], window: int = 0) -> tuple[str, List[Dict]]:
    """
    Merge chunks with enhanced LLM metadata support
    
    Args:
        chunks: List of chunk dictionaries with metadata
        picked: Selected chunk results with LLM scores
        window: Number of adjacent chunks to include (0 for exact chunks only)
        
    Returns:
        Tuple of (stitched_context, enhanced_citations)
    """
    # Collect indices to include (picked + adjacent)
    indices = set()
    for p in picked:
        i = p["idx"]
        for j in range(max(0, i - window), min(len(chunks), i + window + 1)):
            indices.add(j)

    # Build context string (without chunk references)
    stitched_parts = []
    for i in sorted(indices):
        if i < len(chunks):
            chunk_text = chunks[i]["text"]
            stitched_parts.append(chunk_text)
    
    stitched = "\n\n".join(stitched_parts)

    # Enhanced citations with LLM scores and reasoning
    citations = []
    for p in picked:
        chunk_idx = p["idx"]
        
        # Get chunk text preview
        chunk_text = chunks[chunk_idx]["text"]
        text_preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
        
        citation = {
            "chunk_id": chunks[chunk_idx]["id"],
            "filename": chunks[chunk_idx]["filename"],
            "relevance_score": round(p.get("final_score", 0), 3),
            "llm_score": round(p.get("llm_score", 5.0), 1),
            "llm_reasoning": p.get("llm_reasoning", "No reasoning available"),
            "hybrid_score": round(p.get("score", 0), 3),
            "chunk_text": text_preview
        }
        
        citations.append(citation)

    return stitched, citations


def answer_query(user_query: str) -> Dict:
    """
    Main query processing function using multi-stage retrieval with LLM reranking
    """
    # Lightweight routing for greetings/clarifications to avoid unrelated summaries
    intent_label = intent_classifier.classify(user_query)
    normalized = (user_query or "").strip().lower()
    low_information = len(normalized.split()) <= 2 and not any(k in normalized for k in ("who", "what", "where", "when", "why", "how", "list", "compare", "summarize"))
    # Smalltalk patterns (e.g., "how's life", "what's up", "how's <name>'s life")
    smalltalk_pattern = re.compile(r"(what's up|how(?:'s| is) (?:life|it going|everything|things)|how(?:'s| is) [a-z]+(?:'s)? life\b)")
    if intent_label in ("greeting", "clarification") or low_information or smalltalk_pattern.search(normalized):
        guidance_items = [
            "Summarize this document",
            "What are the key points?",
            "List all companies mentioned in the document",
            "Where did the person work?"
        ]
        guidance = "Hi! I can help with your uploaded documents. Try questions like:\n- " + "\n- ".join(guidance_items)
        return {"intent": intent_label, "answer": guidance, "citations": []}

    # Load chunks and embeddings
    chunks, embeddings = load_chunks_and_vectors()
    if not chunks:
        return {"error": "No documents ingested yet."}

    # Low-overlap guard: if query shares almost no content words with corpus, guide user
    def _content_tokens(text: str) -> List[str]:
        # More lenient tokenization - keep words of length 2+ and be less aggressive with stopwords
        tokens = [t for t in normalize(text) if len(t) >= 2 and t not in _STOPWORDS]
        return tokens
    doc_vocab: set = set()
    for c in chunks:
        doc_vocab.update(_content_tokens(c.get("text", "")))
        if len(doc_vocab) > 5000:
            break
    q_tokens = set(_content_tokens(user_query))
    overlap = len(q_tokens & doc_vocab) / (len(q_tokens) or 1)
    if overlap < 0.1 and intent_label in ("kb", "clarification", "greeting"):
        tips = [
            "Summarize this document",
            "What are the key points?",
            "List the companies or organizations mentioned",
            "Where did the person work?"
        ]
        msg = "I couldn't link your question to the document content. Try:\n- " + "\n- ".join(tips)
        return {"intent": intent_label, "answer": msg, "citations": []}

    # Transform query for better matching
    q = transform_query(user_query)
    
    # Perform multi-stage retrieval with LLM reranking
    print(f"\n=== Multi-Stage Retrieval for: '{user_query}' ===")
    results = multistage_retrieval(q, chunks, embeddings)
    
    # Post-process results for context generation
    context, citations = postprocess_multistage_results(chunks, results)

    if context is None:
        # Always attempt LLM: build a minimal context from top-scoring results (ignoring threshold)
        if results:
            try:
                top_idxs = [r["idx"] for r in results[:2]]
                stitched_parts = [chunks[i]["text"] for i in top_idxs if i < len(chunks)]
                context = "\n\n".join(stitched_parts).strip()
                citations = [{
                    "chunk_id": chunks[i]["id"],
                    "filename": chunks[i]["filename"],
                    "relevance_score": round(results[k].get("score", 0), 3)
                } for k, i in enumerate(top_idxs) if i < len(chunks)]
            except Exception:
                context = ""
        # If no results at all but chunks exist, use first few chunks as generic context
        if not context and chunks:
            stitched_parts = [c.get("text", "") for c in chunks[:5] if c.get("text")]
            context = "\n\n".join(stitched_parts).strip()
            citations = []
        # If still empty, return insufficient evidence
        if not context:
            return {"status": "insufficient_evidence", "answer": "I don't have enough relevant data to answer confidently."}

    # Guard: if the stitched context looks like a header/contact block and the
    # user intent is not explicitly a list/comparison/timeline, provide guidance instead
    guard_intent = intent_classifier.classify(user_query)
    # Generic behavior: only trigger the header guard for greetings/clarifications or
    # when the query contains no strong content tokens.
    has_strong_tokens = len([t for t in normalize(user_query) if len(t) >= 4 and t not in _STOPWORDS]) > 0
    if _looks_like_header(context) and guard_intent in ("greeting", "clarification") and not has_strong_tokens:
        tips = [
            "Summarize this document",
            "What are the key points?",
            "List the companies or organizations mentioned",
            "Where did the person work?"
        ]
        msg = "I need a bit more detail. Try questions like:\n- " + "\n- ".join(tips)
        return {"intent": guard_intent, "answer": msg, "citations": []}

    # If context exists but does not share content words with the query,
    # return a clear "not mentioned" instead of summarizing.
    def _content_tokens_local(text: str) -> List[str]:
        return [t for t in normalize(text) if len(t) >= 3 and t not in _STOPWORDS]
    q_tokens_ctx = set(_content_tokens_local(user_query))
    ctx_tokens = set(_content_tokens_local(context or ""))
    if q_tokens_ctx and not (q_tokens_ctx & ctx_tokens):
        return {"intent": intent_label, "answer": "Not mentioned in the provided documents.", "citations": citations}

    # Determine intent (guides formatting only)
    response_type = determine_response_type(user_query)
    # Optionally refine response type with intent
    if intent_label == "list":
        response_type = "a bulleted list with clear, concise items"
    elif intent_label == "comparison":
        response_type = "a structured comparison with clear sections"
    elif intent_label == "timeline":
        response_type = "a chronological timeline or ordered list"
    
    prompt = (
        "You are an AI assistant answering questions based strictly on the provided context. "
        "If the answer isn't present, say 'insufficient evidence'. "
        "Provide clear, well-formatted answers without showing internal chunk references. "
        f"Format your response as: {response_type}\n\n"
        f"Question: {user_query}\n\nContext:\n{context}\n\nAnswer:"
    )

    answer = call_mistral(prompt)
    
    # Check for API errors
    if answer == "API_RATE_LIMIT_ERROR":
        return {"intent": "error", "answer": "Sorry, the AI service is currently experiencing high demand. Please try again in a few minutes.", "citations": citations}
    elif "Error communicating with Mistral API" in answer or "401 Client Error: Unauthorized" in answer:
        # Fallback to structured answer when API is not available
        response_type = determine_response_type(user_query)
        answer = generate_structured_fallback(user_query, context, response_type)
    elif "insufficient evidence" in answer.lower():
        # Prefer explicit "not mentioned" for non-summary questions
        lower_q = (user_query or "").lower()
        asks_summary = any(k in lower_q for k in ("summarize", "summary", "what is this document about"))
        if asks_summary:
            response_type = determine_response_type(user_query)
            answer = generate_structured_fallback(user_query, context, response_type)
        else:
            answer = "Not mentioned in the provided documents."
    
    # hallucination check: does answer mention any unknown facts?
    if "insufficient evidence" not in answer.lower():
        for w in ["http", "image", "table", "source"]:
            if w in answer.lower() and w not in context.lower():
                answer += "\n\n(Note: possible unsupported claim detected.)"

    return {"intent": "kb", "answer": answer, "citations": citations}