#Reads config.yaml

#Calls BM25 (retrieval/bm25_baseline.py) or Vector, then (if reranker: on) calls reranker.

#Returns top-k with normalized scores and a method field.
#DoD: from retrieval.get_contexts import get_contexts works and returns k items.
#Ensure get_contexts.py returns scores (and BM25 max score) so Gen/Safety can threshold for OOS later.
#DoD: each context item includes score (0–1 if normalized) and method.