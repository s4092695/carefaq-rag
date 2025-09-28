# eval/run_retrieval_metrics.py
import argparse, json, math, pathlib
from typing import List
from retrieval.bm25_baseline import search as bm25_search
from retrieval.vector_search import search_vector
from eval.metrics import ndcg_at_k, mean
from eval.utils_eval import load_passages

HERE = pathlib.Path(__file__).parent
DATA = HERE / "datasets" / "questions_known.jsonl"
OUT = HERE / "results" / "retrieval_metrics.md"

def relevance_gains(retrieved_pids: List[str], gold_pids: List[str]) -> List[int]:
    return [1 if pid in set(gold_pids) else 0 for pid in retrieved_pids]

def eval_one(method_name: str, search_fn, k_list=(1,3,5)):
    lines = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    ndcg_rows = {k: [] for k in k_list}
    hit3 = []
    for row in lines:
        q = row["question"]
        gold_pids = row.get("gold_pids", [])
        kmax = max(k_list)
        res = search_fn(q, k=kmax)
        retrieved_pids = [r.get("pid","") for r in res]
        gains = relevance_gains(retrieved_pids, gold_pids)
        for k in k_list:
            ndcg_rows[k].append(ndcg_at_k(gains, k))
        hit3.append(1.0 if any(gains[:3]) else 0.0)
    summary = {f"NDCG@{k}": mean(ndcg_rows[k]) for k in k_list}
    summary["Hit@3"] = mean(hit3)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=[1,3,5])
    args = ap.parse_args()
    k_list = tuple(args.k)

    # BM25
    bm = eval_one("bm25", bm25_search, k_list=k_list)
    # Vector
    ve = eval_one("vector", search_vector, k_list=k_list)

    # Write markdown table
    header = "| Method | " + " | ".join([f"NDCG@{k}" for k in k_list]) + " | Hit@3 |\n"
    header += "|" + " --- |" * (len(k_list)+2) + "\n"
    row_bm = "| BM25 | " + " | ".join([f"{bm[f'NDCG@{k}']:.3f}" for k in k_list]) + f" | {bm['Hit@3']:.3f} |\n"
    row_ve = "| Vector | " + " | ".join([f"{ve[f'NDCG@{k}']:.3f}" for k in k_list]) + f" | {ve['Hit@3']:.3f} |\n"
    OUT.write_text(header + row_bm + row_ve, encoding="utf-8")
    print((header + row_bm + row_ve).strip())

if __name__ == "__main__":
    main()