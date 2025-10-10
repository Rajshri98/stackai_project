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

#TF-IDF scoring
def compute_scores(query_tokens: List[str], docs: List[str]) -> List[Dict]:
    """
    Computes a basic hybrid score = lexical_overlap + tf-idf cosine
    """
    N = len(docs)
    df = {}
    doc_tokens = [normalize(d) for d in docs]
    for toks in doc_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (df[w] + 1)) for w in df}

    # Build query vector
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

        # cosine
        cos = sum(q_vec.get(w, 0) * vec.get(w, 0) for w in q_vec)
        # lexical overlap (Jaccard)
        overlap = len(set(query_tokens) & set(toks)) / (len(set(query_tokens) | set(toks)) or 1)
        score = 0.7 * cos + 0.3 * overlap
        results.append({"idx": idx, "score": score})
    return sorted(results, key=lambda x: x["score"], reverse=True)

#Evidence threshold + citations
def gather_context(chunks: List[str], results: List[Dict], threshold=0.05, top_k=4):
    if not results or results[0]["score"] < threshold:
        return None, []
    context = []
    cites = []
    for r in results[:top_k]:
        context.append(f"[CHUNK {r['idx']}] {chunks[r['idx']]}")
        cites.append({"chunk": r["idx"], "score": round(r["score"], 3)})
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
    context, citations = gather_context(chunks, results)

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