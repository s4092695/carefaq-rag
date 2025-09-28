# eval/run_retrieval_metrics.py
import json, math, pathlib
from retrieval.bm25_baseline import search as bm25_search
from retrieval.vector_search import search_vector
from retrieval.get_contexts import CONFIG

KNOWN = "eval/questions_known.jsonl"
OUT = pathlib.Path("results/final_metrics.md")

def ndcg_at_k(rel, k):
    dcg = sum((g / math.log2(i+2)) for i,g in enumerate(rel[:k]))
    idcg = sum((g / math.log2(i+2)) for i,g in enumerate(sorted(rel, reverse=True)[:k]))
    return 0.0 if idcg == 0 else dcg/idcg

def _eval(path, method, k_vals=(1,3,5)):
    qs = [json.loads(l) for l in open(path, encoding="utf-8")]
    sums = {k:0.0 for k in k_vals}; hit3 = 0
    for q in qs:
        if method == "bm25":
            res = bm25_search(q["question"], 5)
        elif method == "vector":
            res = search_vector(q["question"], 5)
        else:  # vector + rerank (use get_contexts with reranker=on)
            from retrieval.get_contexts import get_contexts
            res = get_contexts(q["question"], 5)

        ids = [r["pid"] for r in res]
        rel = [1 if any(g in ids[i] for g in q["gold_pids"]) else 0 for i in range(len(ids))]
        for K in k_vals: sums[K] += ndcg_at_k(rel, K)
        if any(g in ids[:3] for g in q["gold_pids"]): hit3 += 1
    n = max(1, len(qs))
    return {f"NDCG@{K}": round(sums[K]/n,3) for K in k_vals} | {"Top3_Hit": f"{round(100*hit3/n)}%", "Q": n}

if __name__ == "__main__":
    bm25 = _eval(KNOWN, "bm25")
    vec  = _eval(KNOWN, "vector")
    # optional rerank row if config says on
    rr = None
    if CONFIG.get("reranker","off") == "on":
        rr = _eval(KNOWN, "vector+rerank")

    table = (
      "| Method | NDCG@1 | NDCG@3 | NDCG@5 | Top-3 Hit | Q |\n"
      "|---|---:|---:|---:|---:|---:|\n"
      f"| BM25          | {bm25['NDCG@1']} | {bm25['NDCG@3']} | {bm25['NDCG@5']} | {bm25['Top3_Hit']} | {bm25['Q']} |\n"
      f"| Vector (bge)  | {vec['NDCG@1']} | {vec['NDCG@3']} | {vec['NDCG@5']} | {vec['Top3_Hit']} | {vec['Q']} |\n"
    )
    if rr:
        table += f"| Vector + Rerank | {rr['NDCG@1']} | {rr['NDCG@3']} | {rr['NDCG@5']} | {rr['Top3_Hit']} | {rr['Q']} |\n"

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(table, encoding="utf-8")
    print(table)
