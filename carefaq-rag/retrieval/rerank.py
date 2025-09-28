# retrieval/get_contexts.py
import json, yaml
from pathlib import Path
from typing import List, Dict

# required: retrieval/bm25_baseline.py must expose search(q, k)
from retrieval.bm25_baseline import search as bm25_search
from retrieval.vector_search import search_vector

CONFIG = {
    "chunk_size": 600,
    "overlap": 80,
    "top_k": 5,
    "retriever": "vector",     # bm25 | vector
    "reranker": "off",         # on | off
    "emb_model": "BAAI/bge-small-en-v1.5"
}

CFG_PATH = Path("retrieval/config.yaml")
if CFG_PATH.exists():
    CONFIG.update(yaml.safe_load(open(CFG_PATH)))

def _maybe_rerank(query: str, cxts: List[Dict]):
    if CONFIG.get("reranker", "off") != "on":
        return cxts
    try:
        from retrieval.rerank import rerank
        # Ensure candidate texts exist
        cand = [{"text": c["text"], **c} for c in cxts]
        return rerank(query, cand)
    except Exception:
        # fail open: keep original order
        return cxts

def get_contexts(query: str, k: int = None) -> List[Dict]:
    k = k or CONFIG.get("top_k", 5)
    if CONFIG.get("retriever") == "bm25":
        cxts = bm25_search(query, k)
        # bm25 script may not include text; hydrate if needed
        # Optional: add a short text fetch here if you want.
        for c in cxts: c.setdefault("method", "bm25")
    else:
        cxts = search_vector(query, k)
    return _maybe_rerank(query, cxts)[:k]
