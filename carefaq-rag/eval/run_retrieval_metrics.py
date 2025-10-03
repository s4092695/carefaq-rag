# eval/run_retrieval_metrics.py
import argparse, json, pathlib
from typing import List, Dict, Callable

from eval.metrics import ndcg_at_k, mean
from retrieval.bm25_baseline import search as bm25_search
from retrieval.vector_search import search_vector
from retrieval.hybrid import search_hybrid
from retrieval.get_contexts import get_contexts  # for *+RR methods

HERE = pathlib.Path(__file__).parent
DEFAULT_DATASET = HERE / "datasets" / "questions_known.jsonl"
OUT = HERE / "results" / "retrieval_metrics.md"

# All methods return List[Dict] with at least a 'pid' field
METHOD_FNS: Dict[str, Callable[[str, int], List[Dict]]] = {
    "BM25":       lambda q, k: bm25_search(q, k),
    "Vector":     lambda q, k: search_vector(q, k),
    "Hybrid":     lambda q, k: search_hybrid(q, k),
    "Vector+RR":  lambda q, k: get_contexts(q, k=k, method="vector_rr"),
    "Hybrid+RR":  lambda q, k: get_contexts(q, k=k, method="hybrid_rr"),
}

def relevance_gains(retrieved_pids: List[str], gold_pids: List[str]) -> List[int]:
    gold = set(gold_pids)
    return [1 if pid in gold else 0 for pid in retrieved_pids]

def eval_one(search_fn, dataset_path: pathlib.Path, k_list=(1, 3, 5)):
    lines = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ndcg_rows = {k: [] for k in k_list}
    hit3 = []
    for row in lines:
        q = row["question"]
        gold_pids = row.get("gold_pids", [])
        kmax = max(k_list)
        res = search_fn(q, k=kmax)  # -> list[dict]
        retrieved_pids = [r.get("pid", "") for r in res]
        gains = relevance_gains(retrieved_pids, gold_pids)
        for k in k_list:
            ndcg_rows[k].append(ndcg_at_k(gains, k))
        hit3.append(1.0 if any(gains[:3]) else 0.0)
    summary = {f"NDCG@{k}": mean(ndcg_rows[k]) for k in k_list}
    summary["Hit@3"] = mean(hit3)
    return summary

def rank_of_gold(pred_pids: List[str], gold: List[str]):
    g = set(gold)
    for i, pid in enumerate(pred_pids, start=1):
        if pid in g:
            return i
    return None

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to questions_known*.jsonl")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5], help="List of cutoffs, e.g., --k 3 5")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["BM25", "Vector", "Hybrid", "Vector+RR", "Hybrid+RR"],
        choices=list(METHOD_FNS.keys()),
        help="Which methods to evaluate",
    )
    ap.add_argument("--dump-ranks", action="store_true", help="Print per-question gold ranks for each method")
    args = ap.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    k_list = tuple(args.k)
    methods = args.methods

    # Evaluate
    summaries = {name: eval_one(METHOD_FNS[name], dataset_path, k_list=k_list) for name in methods}

    # Markdown table
    header = "| Method | " + " | ".join([f"NDCG@{k}" for k in k_list]) + " | Hit@3 |\n"
    header += "|" + " --- |" * (len(k_list) + 2) + "\n"
    rows_md = []
    for name in methods:
        s = summaries[name]
        ndcg_cells = [f"{s[f'NDCG@{k}']:.3f}" for k in k_list]
        row = f"| {name} | " + " | ".join(ndcg_cells) + f" | {s['Hit@3']:.3f} |\n"
        rows_md.append(row)

    md = header + "".join(rows_md)
    OUT.write_text(md, encoding="utf-8")
    print(md.strip())

    # Optional per-question ranks
    if args.dump_ranks:
        lines = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        kmax = max(k_list)
        print(f"\nPer-question ranks (k cutoff = {kmax}):")
        for row in lines:
            qid, gold = row.get("qid", "<noid>"), row.get("gold_pids", [])
            line = [f"{qid} gold={gold}"]
            for name in methods:
                preds = [r.get("pid", "") for r in METHOD_FNS[name](row["question"], k=kmax)]
                r = rank_of_gold(preds, gold)
                line.append(f"{name}: {r if r is not None else 'miss'}")
            print(" | ".join(line))

if __name__ == "__main__":
    main()
