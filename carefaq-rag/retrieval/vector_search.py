# retrieval/vector_search.py
import os, json, numpy as np
from pathlib import Path

# Embeddings: small, CPU-friendly
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

try:
    import faiss
    USE_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    USE_FAISS = False

from sentence_transformers import SentenceTransformer

# --- load corpus ---
PASSAGES_PATH = Path("kb/chunks/passages.jsonl")
if not PASSAGES_PATH.exists():
    raise FileNotFoundError("kb/chunks/passages.jsonl not found. Run: python scripts/make_chunks.py")

passages = [json.loads(l) for l in open(PASSAGES_PATH, encoding="utf-8")]
texts = [p["text"] for p in passages]

# --- embed corpus ---
_embedder = SentenceTransformer(EMB_MODEL)
X = np.asarray(_embedder.encode(texts, normalize_embeddings=True), dtype="float32")

# --- build index ---
if USE_FAISS:
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
else:
    index = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X)

def _similarity_search(qv: np.ndarray, k: int):
    if USE_FAISS:
        D, I = index.search(qv, k)
        sims, idxs = D[0], I[0]    # higher is better
    else:
        dists, idxs = index.kneighbors(qv, n_neighbors=k, return_distance=True)
        sims, idxs = (1.0 - dists[0]), idxs[0]   # convert distance->similarity
    return sims, idxs

def search_vector(query: str, k: int = 5):
    qv = np.asarray(_embedder.encode([query], normalize_embeddings=True), dtype="float32")
    sims, idxs = _similarity_search(qv, k)
    out = []
    for i, s in zip(idxs, sims):
        p = passages[int(i)]
        out.append({
            "pid": p["pid"], "source_url": p["source_url"],
            "text": p["text"], "score": float(s), "method": "vector"
        })
    return out
