# retrieval/vector_search.py
import os, json, re, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this question for retrieving relevant documents: "

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "kb" / "index"
PASSAGES_PATH = ROOT / "kb" / "chunks" / "passages.jsonl"

# ----- load passages & index -----
_passages = {}
for l in PASSAGES_PATH.read_text(encoding="utf-8").splitlines():
    if not l.strip(): continue
    j = json.loads(l); _passages[j["pid"]] = j

_pids = json.loads((INDEX_DIR / "pids.json").read_text(encoding="utf-8"))
_embs = np.load(INDEX_DIR / "embeddings.npy").astype("float32")

# backend
try:
    import faiss
    _index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    _backend = "faiss"
except Exception:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine", n_neighbors=min(50, len(_embs)))
    nn.fit(_embs)
    _index = nn
    _backend = "sklearn"

_embedder = SentenceTransformer(EMB_MODEL)

# ----- light alias expansion (mirrors BM25 idea) -----
_aliases = [
    (re.compile(r"\breferral(s)?\b", flags=re.I),
     " referral referrals refer referal GP doctor walk-in centre walk in centre WIC "),
    (re.compile(r"\bwalk[- ]?in\b", flags=re.I),
     " walk-in walk in WIC centre clinic "),
    (re.compile(r"\baccessibil(it|ty)\b", flags=re.I),
     " accessibility accessible access "),
]

def _expand_query(q: str) -> str:
    out = q
    for pat, add in _aliases:
        if pat.search(out):
            out += " " + add
    return out

def _similarity_search(qv: np.ndarray, k: int, pool_k: int):
    if _backend == "faiss":
        sims, idxs = _index.search(qv, pool_k)  # larger pool, we trim later
        return sims[0], idxs[0][:k], idxs[0]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", n_neighbors=min(pool_k, len(_embs)))
        nn.fit(_embs)
        dists, idxs = nn.kneighbors(qv, n_neighbors=min(pool_k, len(_embs)))
        sims = 1.0 - dists[0]
        return sims, idxs[0][:k], idxs[0]

def search_vector(query: str, k: int = 5, pool_k: int = 60):
    q_text = QUERY_PREFIX + _expand_query(query)
    qv = _embedder.encode([q_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sims, topk_idxs, _ = _similarity_search(qv, k=k, pool_k=pool_k)
    out = []
    for i, s in zip(topk_idxs, sims[:len(topk_idxs)]):
        pid = _pids[int(i)]
        p = _passages.get(pid, {"pid": pid, "source_url": "", "text": ""})
        out.append({
            "pid": pid,
            "source_url": p.get("source_url", ""),
            "text": p.get("text", ""),
            "score": float(s),
            "method": "vector",
        })
    return out

# Handy helper for a bigger candidate pool (used by reranker callers)
def search_vector_candidates(query: str, pool_k: int = 60):
    q_text = QUERY_PREFIX + _expand_query(query)
    qv = _embedder.encode([q_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if _backend == "faiss":
        sims, idxs = _index.search(qv, pool_k)
        idxs = idxs[0]; sims = sims[0]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", n_neighbors=min(pool_k, len(_embs)))
        nn.fit(_embs)
        dists, idxs = nn.kneighbors(qv, n_neighbors=min(pool_k, len(_embs)))
        idxs = idxs[0]; sims = 1.0 - dists[0]
    items = []
    for i, s in zip(idxs, sims):
        pid = _pids[int(i)]
        p = _passages.get(pid, {"pid": pid, "source_url": "", "text": ""})
        items.append({"pid": pid, "source_url": p.get("source_url",""), "text": p.get("text",""),
                      "score": float(s), "method": "vector"})
    return items
