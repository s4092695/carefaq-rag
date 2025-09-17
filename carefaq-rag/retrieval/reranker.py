#Add retrieval/rerank.py using a cross-encoder (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to rerank top-k.

#Wire it via config.yaml (reranker: on), then re-run metrics to add a third row.
#DoD: table shows “Vector+Rerank”; aim for ΔNDCG@3 ≥ +0.05 vs BM25 on known Qs.