# retrieval/reranker.py
from typing import List, Dict, Tuple
import os
from sentence_transformers import CrossEncoder

# Choose via env; default to SMALL for speed
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model = None
def _get_model():
    global _model
    if _model is None:
        # max_length trims tokenized pairs; helps a lot on CPU
        _model = CrossEncoder(RERANKER_MODEL, max_length=int(os.getenv("RERANKER_MAX_LEN", "256")))
    return _model

def rerank(query: str, items: List[Dict], top_k: int = 5) -> List[Dict]:
    if not items:
        return items
    model = _get_model()
    pairs = [(query, it.get("text", "")) for it in items]
    scores = model.predict(pairs)  # CPU ok with MiniLM
    out = []
    for s, it in zip(scores, items):
        it2 = dict(it)
        it2["method"] = (it.get("method") or "") + "+rr"
        it2["score"] = float(s)
        out.append(it2)
    out.sort(key=lambda x: -x["score"])
    return out[:top_k]
