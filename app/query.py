# Handles query understanding, search, reranking, and answer generation

import os, re, json, math, requests
from collections import Counter
from typing import List, Dict
from app.store import load_chunks

# Detect intent
def needs_search(query: str) -> bool:
    """
    Returns False for greetings / chit-chat queries like 'hi', 'thanks'.
    """
    query = query.lower().strip()
    if query in ["hi", "hello", "hey", "thanks", "thank you", "goodbye"]:
        return False
    return len(query.split()) > 2

# Normalize and Transform
def normalize(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t]

def transform_query(query: str) -> str:
    q = query.lower()
    q = re.sub(r"\bwhat\s+is\b", "definition of", q)
    q = re.sub(r"\bhow\s+to\b", "steps to", q)
    return q.strip()

# Hybrid Semantic + Keyword Search
def compute_scores(query_tokens: List[str], docs: List[str]) -> List[Dict]:
    """
    Combines lexical (keyword) and semantic similarity using TF-IDF-style weighting.
    Returns ranked document chunks with combined scores.
    """
    N = len(docs)
    df = {}
    doc_tokens = [normalize(d) for d in docs]

    # Document Frequency
    for toks in doc_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (df[w] + 1)) for w in df}

    # Build query vector (TF-IDF)
    q_tf = Counter(query_tokens)
    q_vec = {w: (1 + math.log(c)) * idf.get(w, 0) for w, c in q_tf.items()}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    q_vec = {k: v / q_norm for k, v in q_vec.items()}

    results = []
    for idx, toks in enumerate(doc_tokens):
        tf = Counter(toks)
        vec = {w: (1 + math.log(c)) * idf.get(w, 0) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vec = {k: v / norm for k, v in vec.items()}

        # Semantic similarity (cosine between TF-IDF vectors)
        cosine_sim = sum(q_vec.get(w, 0) * vec.get(w, 0) for w in q_vec)

        # Lexical overlap (Jaccard index)
        overlap = len(set(query_tokens) & set(toks)) / (len(set(query_tokens) | set(toks)) or 1)

        # Combine scores with tunable weights
        combined_score = 0.6 * cosine_sim + 0.4 * overlap

        results.append({
            "idx": idx,
            "semantic_score": round(cosine_sim, 4),
            "keyword_score": round(overlap, 4),
            "score": round(combined_score, 4)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

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

#Call Mistral LLM
def call_mistral(prompt: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        return "Mistral API key missing."
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


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


def postprocess_results(query_tokens: List[str], chunks: List[str], results: List[Dict],
                        min_threshold: float = 0.06, final_k: int = 4) -> (str, List[Dict]):
    # 1) Fuse semantic+keyword ranks; 2) Select diverse top-k; 3) Merge neighbors; 4) Enforce evidence threshold
    if not results or results[0]["score"] < min_threshold:
        return None, []
    fused = _fuse_by_rrf(results)
    picked = _mmr_select(query_tokens, fused[:max(10, final_k*2)], chunks, lambda_balance=0.7, top_k=final_k)
    stitched, cites = _stitch_adjacent(chunks, picked, window=1)
    if not stitched.strip():
        return None, []
    return stitched, cites


#Full pipeline
def answer_query(user_query: str) -> Dict:
    if not needs_search(user_query):
        return {"intent": "non_info", "answer": "Hi! Please ask a question about your uploaded files."}

    chunks = load_chunks()
    if not chunks:
        return {"error": "No documents ingested yet."}

    q = transform_query(user_query)
    q_tokens = normalize(q)
    results = compute_scores(q_tokens, chunks)
    context, citations = postprocess_results(q_tokens, chunks, results)

    if context is None:
        return {"status": "insufficient_evidence", "answer": "I don't have enough data to answer confidently."}

    prompt = (
        "You are an AI assistant answering questions based strictly on the provided context. "
        "If the answer isn't present, say 'insufficient evidence'. Include chunk IDs in brackets.\n\n"
        f"Question: {user_query}\n\nContext:\n{context}\n\nAnswer:"
    )

    answer = call_mistral(prompt)
    # simple hallucination check: does answer mention any unknown facts?
    if "insufficient evidence" not in answer.lower():
        for w in ["http", "image", "table", "source"]:
            if w in answer.lower() and w not in context.lower():
                answer += "\n\n(Note: possible unsupported claim detected.)"

    return {"intent": "kb", "answer": answer, "citations": citations}