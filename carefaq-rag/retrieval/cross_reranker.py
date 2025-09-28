# retrieval/cross_reranker.py
from typing import List, Dict
from sentence_transformers import CrossEncoder

# Light, CPU-friendly cross-encoder
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce = None

def get_model():
    global _ce
    if _ce is None:
        _ce = CrossEncoder(MODEL_NAME)
    return _ce

def rerank(query: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
    if not contexts:
        return []
    ce = get_model()
    pairs = [(query, c.get("text","")) for c in contexts]
    scores = ce.predict(pairs)
    ranked = sorted(zip(contexts, scores), key=lambda x: -float(x[1]))
    out = []
    for c, s in ranked[:top_k]:
        c2 = dict(c)
        c2["score"] = float(s)
        c2["method"] = c.get("method","vector") + "+rerank"
        out.append(c2)
    return out