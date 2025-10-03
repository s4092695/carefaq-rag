from .bm25_baseline import search as bm25_search
from .vector_search import search_vector, search_vector_candidates
from .hybrid import search_hybrid
from .reranker import rerank

def get_contexts(question: str, k: int = 5, method: str = "hybrid"):
    if method == "bm25":
        return bm25_search(question, k=k)
    if method == "vector":
        return search_vector(question, k=k)  # top-k only
    if method == "vector_rr":
        return rerank(question, search_vector_candidates(question, pool_k=60), top_k=k)
    if method == "hybrid":
        return search_hybrid(question, k=k)
    if method == "hybrid_rr":
        # hybrid() already pulls a big pool internally
        return rerank(question, search_hybrid(question, k=60), top_k=k)
    return search_hybrid(question, k=k)
