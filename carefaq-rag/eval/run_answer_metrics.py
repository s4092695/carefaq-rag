# eval/run_answer_metrics.py
import argparse, csv, json, pathlib, time
from typing import Dict, List
from eval.metrics import token_f1, mean
from eval.utils_eval import load_passages

from gen.rag_chain import answer as rag_answer
from retrieval.get_contexts import CONFIG as RETR_CONFIG

HERE = pathlib.Path(__file__).parent
DATA = HERE / "datasets" / "gold_qa.csv"
OUT_CSV = HERE / "results" / "answer_metrics.csv"
OUT_MD  = HERE / "results" / "answer_metrics.md"

def set_retriever(retriever: str, reranker: str, k: int):
    RETR_CONFIG["retriever"] = retriever  # "bm25" | "vector"
    RETR_CONFIG["reranker"] = reranker    # "on" | "off"
    RETR_CONFIG["top_k"] = int(k)

def is_noanswer(output: Dict) -> bool:
    # Our chain uses method="noanswer" for low confidence
    return (output.get("method") == "noanswer") or (not output.get("sources"))

def load_gold_rows(path: pathlib.Path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def citations_supported(ans_text: str, cited_urls: List[str], by_url) -> bool:
    ans_l = (ans_text or "").strip().lower()
    if not ans_l:
        return False
    for url in cited_urls:
        for p in by_url.get(url, []):
            if ans_l in (p.get("text","").lower()):
                return True
    return False

def gold_supported(gold_text: str, cited_urls: List[str], by_url) -> bool:
    gold_l = (gold_text or "").strip().lower()
    if not gold_l:
        return False
    for url in cited_urls:
        for p in by_url.get(url, []):
            if gold_l in (p.get("text","").lower()):
                return True
    return False

def allowed_url_check(cited_urls: List[str], allowed_csv: str) -> bool:
    if not allowed_csv:
        return True
    allowed = [u.strip() for u in allowed_csv.split(",") if u.strip()]
    if not allowed:
        return True
    return all((u in allowed) for u in cited_urls if u)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["bm25","vector"], default="vector")
    ap.add_argument("--reranker", choices=["on","off"], default="off")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    set_retriever(args.retriever, args.reranker, args.k)
    passages, by_pid, by_url = load_passages()

    gold_rows = load_gold_rows(DATA)

    metrics = {
        "n": 0,
        "percent_unanswered": 0.0,
        "accuracy_f1": 0.0,
        "faithfulness": 0.0,     # system answer text supported by any cited passage
        "source_correctness": 0.0,  # gold or system answer appears in cited passage AND urls allowed
        "latency_ms": 0.0,
    }

    # fairness by tag buckets
    per_tag = {}

    detailed_rows = []
    for row in gold_rows:
        qid = row["qid"]
        q = row["question"]
        gold = row["gold_answer"]
        tag = row.get("tag","")
        allowed = row.get("allowed_urls","")

        t0 = time.time()
        out = rag_answer(q, k=args.k)
        dt = (time.time() - t0) * 1000

        sys_ans = out.get("answer","")
        sys_cites = [u for u in out.get("sources",[]) if u]
        noans = is_noanswer(out)

        # % unanswered: if gold empty -> expect no-answer; if gold non-empty -> expect answer
        gold_is_unanswer = (not gold) or (gold.strip() == "")
        ok_unanswered = (gold_is_unanswer and noans)
        wrong_unanswered = (gold_is_unanswer and not noans)

        # accuracy: token F1 vs gold (only when gold provided)
        acc_f1 = token_f1(sys_ans, gold) if not gold_is_unanswer else 0.0

        faithful = citations_supported(sys_ans, sys_cites, by_url)
        source_ok = (gold_supported(gold, sys_cites, by_url) or faithful) and allowed_url_check(sys_cites, allowed)

        detailed_rows.append({
            "qid": qid, "tag": tag, "question": q,
            "gold_answer": gold, "sys_answer": sys_ans,
            "noanswer": int(noans), "ok_unanswered": int(ok_unanswered), "wrong_unanswered": int(wrong_unanswered),
            "f1": acc_f1, "faithful": int(faithful), "source_correct": int(source_ok),
            "latency_ms": round(dt, 1), "cited_urls": ";".join(sys_cites)
        })

        bucket = per_tag.setdefault(tag or "untagged", {"n":0, "f1":[],"faith":[],"src_ok":[]})
        bucket["n"] += 1
        if not gold_is_unanswer:
            bucket["f1"].append(acc_f1)
        bucket["faith"].append(1.0 if faithful else 0.0)
        bucket["src_ok"].append(1.0 if source_ok else 0.0)

        # aggregate
        metrics["n"] += 1
        if ok_unanswered: metrics["percent_unanswered"] += 1.0
        if not gold_is_unanswer: metrics["accuracy_f1"] += acc_f1
        if faithful: metrics["faithfulness"] += 1.0
        if source_ok: metrics["source_correctness"] += 1.0
        metrics["latency_ms"] += dt

    n = metrics["n"] or 1
    # normalize
    metrics["percent_unanswered"] = 100.0 * metrics["percent_unanswered"] / n
    # average accuracy only over answerable qs
    num_answerable = sum(1 for r in detailed_rows if r["gold_answer"])
    metrics["accuracy_f1"] = (metrics["accuracy_f1"] / max(1, num_answerable))
    metrics["faithfulness"] = (metrics["faithfulness"] / n)
    metrics["source_correctness"] = (metrics["source_correctness"] / n)
    metrics["latency_ms"] = (metrics["latency_ms"] / n)

    # write CSV of detailed rows
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()) if detailed_rows else ["qid"])
        w.writeheader()
        for r in detailed_rows:
            w.writerow(r)

    # write summary markdown
    lines = []
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| % Unanswered (overall) | {metrics['percent_unanswered']:.1f}% |")
    lines.append(f"| Accuracy (token F1, answerable only) | {metrics['accuracy_f1']:.3f} |")
    lines.append(f"| Faithfulness (supported by cited passage) | {metrics['faithfulness']:.3f} |")
    lines.append(f"| Source correctness (gold/system supported & URLs ok) | {metrics['source_correctness']:.3f} |")
    lines.append(f"| Avg latency (ms) | {metrics['latency_ms']:.1f} |")
    lines.append("")
    lines.append("### Fairness by tag")
    lines.append("| Tag | n | Avg F1 (answ.) | Faithfulness | Source correctness |")
    lines.append("|---|---:|---:|---:|---:|")
    for tag, b in sorted(per_tag.items()):
        lines.append(f"| {tag} | {b['n']} | { (sum(b['f1'])/max(1,len(b['f1']))):.3f} | { (sum(b['faith'])/b['n']):.3f} | { (sum(b['src_ok'])/b['n']):.3f} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()