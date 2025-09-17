#Load kb/chunks/passages.jsonl

#Embed with emb_model (sentence-transformers)

#Build FAISS IndexFlatIP (with sklearn NearestNeighbors as fallback)

#search_vector(q,k) → list of {pid, source_url, text, score}
#DoD: python -c "from retrieval.vector_search import search_vector; print(search_vector('bulk bill',3))" prints results.