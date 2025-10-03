# retrieval/hybrid.py
from typing import List, Dict
import re
from .bm25_baseline import search as bm25_search
from .vector_search import search_vector_candidates

W_VEC = 3.0
W_BM25 = 1.0
K_RRF = 60
POOL_K = 60
TITLE_MATCH_BONUS = 0.25

_word = re.compile(r"[A-Za-z0-9]+")
def _tok(s: str): return [w.lower() for w in _word.findall(s or "")]

def _title_match_bonus(item: Dict, q_tokens):
    txt = item.get("text", "")
    title = ""
    if txt.startswith("[TITLE]"):
        try: title = txt.split("] ", 1)[1].split(". ", 1)[0]
        except Exception: title = ""
    title_tokens = set(_tok(title))
    return TITLE_MATCH_BONUS if title_tokens.intersection(q_tokens) else 0.0

def _weighted_rrf(bm: List[Dict], ve: List[Dict], q: str, k: int) -> List[Dict]:
    q_tokens = _tok(q)
    scores, best = {}, {}
    bm_rank = {it["pid"]: r for r, it in enumerate(bm, start=1)}
    ve_rank = {it["pid"]: r for r, it in enumerate(ve, start=1)}
    for pid in (set(bm_rank) | set(ve_rank)):
        r_bm = bm_rank.get(pid); r_ve = ve_rank.get(pid)
        s = 0.0
        if r_bm is not None: s += W_BM25 * (1.0 / (K_RRF + r_bm))
        if r_ve is not None: s += W_VEC  * (1.0 / (K_RRF + r_ve))
        it = next((x for x in ve if x["pid"] == pid), None) or next((x for x in bm if x["pid"] == pid), None)
        s += _title_match_bonus(it, q_tokens)
        scores[pid] = s
        if pid not in best or float(it.get("score", 0)) > float(best[pid].get("score", 0)):
            best[pid] = it
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [{**best[pid], "method": "hybrid_rrf_w", "score": float(scores[pid])} for pid, _ in ranked]

def search_hybrid(query: str, k: int = 5, pool_k: int = POOL_K) -> List[Dict]:
    bm = bm25_search(query, k=pool_k)
    ve = search_vector_candidates(query, pool_k=pool_k)
    return _weighted_rrf(bm, ve, query, k)
