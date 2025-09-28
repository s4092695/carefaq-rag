# Evaluation (CareFAQ RAG)

This folder contains **two** complementary evaluations:

1. **Retrieval quality** (`run_retrieval_metrics.py`): reports NDCG@1/3/5 and Hit@3 against *known-answer* queries where we provide a gold passage id (pid).
2. **Answer quality** (`run_answer_metrics.py`): reports **% Unanswered** (Walert-style), **Accuracy** (token-F1 against gold answer), **Faithfulness/Attribution** (is the answer supported by a cited passage), and per-topic **Fairness** breakdowns.

## Datasets

- `datasets/questions_known.jsonl`: each line is `{"qid","question","gold_pids":[...], "tag"}` for retrieval NDCG/Hit.
- `datasets/gold_qa.csv`: columns: `qid,question,gold_answer,tag,allowed_urls` (last column optional comma-separated whitelist).

Start with the provided examples and expand to ~20–50 queries across key topics (fees/hours/referrals/telehealth/etc.).

## How to run

### 1) Retrieval metrics (BM25 vs Vector vs Vector+Rerank)
```bash
# from repo root or from carefaq-rag/
python eval/run_retrieval_metrics.py --k 3 5
```

### 2) Answer metrics (+ % Unanswered)
```bash
python eval/run_answer_metrics.py --k 5 --retriever vector --reranker off
python eval/run_answer_metrics.py --k 5 --retriever bm25  --reranker off
python eval/run_answer_metrics.py --k 5 --retriever vector --reranker on
```

Outputs are written under `eval/results/` as CSV + Markdown.

## Notes

- The eval scripts monkey-patch `retrieval.get_contexts.CONFIG` so you can switch retriever/reranker without editing code.
- *No-answer* detection relies on the `method` field from `gen/rag_chain.answer` (it returns `"noanswer"` when confidence is low).
- Faithfulness checks whether the system’s answer phrase appears in any **cited** passage text; Source correctness checks that either the gold answer text or the system answer appears in a cited passage *and* (optionally) that cited URLs are whitelisted in `allowed_urls`.