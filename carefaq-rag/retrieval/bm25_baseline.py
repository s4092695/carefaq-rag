import json, sys
from rank_bm25 import BM25Okapi
passages = [json.loads(l) for l in open("kb/chunks/passages.jsonl", encoding="utf-8")]
docs = [p["text"] for p in passages]
tok = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tok)
def search(q, k=5):
    scores = bm25.get_scores(q.lower().split())
    idx = range(len(scores))
    top = sorted(zip(idx, scores), key=lambda x: -x[1])[:k]
    return [{"pid": passages[i]["pid"], "source_url": passages[i]["source_url"], "score": float(s)} for i,s in top]
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the clinic opening hours?"
    for r in search(q, k=5):
        print(r)
