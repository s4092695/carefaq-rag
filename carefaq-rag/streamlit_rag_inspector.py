
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy deps wrapped in try/except so app can still boot
try:
    import ollama  # pip install ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

# Embedding-based "fast" judge
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMB = True
except Exception:
    HAS_EMB = False

# Try to import user repo retrieval/rerank utilities (adjust these if your project paths differ)
REPO_IMPORT_OK = True
try:
    # Example imports; change to match your project structure
    from retrieval.vector_search import search_vector as _search_vector_internal
    from retrieval.hybrid import search_hybrid as _search_hybrid_internal
    from retrieval.get_contexts import rerank as _rerank_internal
except Exception:
    REPO_IMPORT_OK = False

st.set_page_config(page_title="RAG Insight Console", layout="wide")

# -----------------------------
# Utility: load passages & datasets
# -----------------------------
def load_passages(passages_path: Path) -> Dict[str, Dict]:
    """Load passages.jsonl -> dict pid -> record{pid, text, doc_id?}"""
    pid2rec = {}
    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            pid2rec[j["pid"]] = j
    return pid2rec

def load_questions(jsonl_path: Path) -> List[Dict]:
    out = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

def pid_to_docid(pid: str, pid2rec: Dict[str, Dict]) -> str:
    if pid in pid2rec and "doc_id" in pid2rec[pid]:
        return pid2rec[pid]["doc_id"]
    # Fallback heuristic: prefix before underscore
    return pid.split("_")[0] if "_" in pid else pid

# -----------------------------
# Retrieval adapters
# -----------------------------
def search_vector(query: str, k: int = 5, pool_k: int = 200) -> List[Tuple[str, float]]:
    """
    Returns a list of (pid, score) in DESC order.
    Prefers your repo search if import succeeded; otherwise, tries simple embedding-based search.
    """
    if REPO_IMPORT_OK:
        # Use internal retrieval that returns pids; if it returns records, adapt below.
        try:
            results = _search_vector_internal(query, k=k, pool_k=pool_k)
            # Normalize to list[(pid, score)]
            out = []
            for r in results:
                if isinstance(r, dict) and "pid" in r:
                    out.append((r["pid"], r.get("score", 0.0)))
                elif isinstance(r, (list, tuple)) and len(r) >= 1:
                    out.append((r[0], float(r[1]) if len(r) > 1 else 0.0))
                else:
                    out.append((str(r), 0.0))
            return out[:k]
        except Exception as e:
            st.warning(f"Vector search (repo) failed: {e}. Falling back to embedding search.")
    # Fallback: embedding-based cosine search over entire corpus (slow for large KB)
    return embed_search(query, k)

def search_hybrid(query: str, k: int = 5, pool_k: int = 200) -> List[Tuple[str, float]]:
    if REPO_IMPORT_OK:
        try:
            results = _search_hybrid_internal(query, k=pool_k)  # pull a big pool
            # Normalize
            out = []
            for r in results:
                if isinstance(r, dict) and "pid" in r:
                    out.append((r["pid"], r.get("score", 0.0)))
                elif isinstance(r, (list, tuple)) and len(r) >= 1:
                    out.append((r[0], float(r[1]) if len(r) > 1 else 0.0))
                else:
                    out.append((str(r), 0.0))
            # Truncate to k
            return out[:k]
        except Exception as e:
            st.warning(f"Hybrid search (repo) failed: {e}. Falling back to vector.")
    return search_vector(query, k=k, pool_k=pool_k)

def rerank(query: str, candidate_pids: List[str], pid2rec: Dict[str, Dict], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    If your repo reranker is available, use it.
    Otherwise, do a simple embedding-based rerank by question->chunk cosine.
    """
    if REPO_IMPORT_OK:
        try:
            # The internal rerank likely expects candidate dicts; adapt here:
            # Build candidate dicts with 'pid' and 'text'
            candidates = [{"pid": pid, "text": pid2rec[pid]["text"]} for pid in candidate_pids if pid in pid2rec]
            results = _rerank_internal(query, candidates, top_k=top_k)
            out = []
            for r in results:
                if isinstance(r, dict) and "pid" in r:
                    out.append((r["pid"], r.get("score", 0.0)))
                elif isinstance(r, (list, tuple)) and len(r) >= 1:
                    out.append((r[0], float(r[1]) if len(r) > 1 else 0.0))
                else:
                    out.append((str(r), 0.0))
            return out[:top_k]
        except Exception as e:
            st.warning(f"Repo rerank failed: {e}. Falling back to embedding rerank.")
    # Fallback: embed-based rerank
    if not HAS_EMB:
        st.error("Embedding rerank requires sentence-transformers. Please install it.")
        return [(pid, 0.0) for pid in candidate_pids[:top_k]]
    model = get_emb_model()
    qv = model.encode([query])
    vecs = model.encode([pid2rec[pid]["text"] for pid in candidate_pids])
    sims = cosine_similarity(qv, vecs)[0]
    order = np.argsort(-sims)
    return [(candidate_pids[i], float(sims[i])) for i in order[:top_k]]

# -----------------------------
# Embedding search (fallback)
# -----------------------------
@st.cache_resource
def get_emb_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_search(query: str, k: int) -> List[Tuple[str, float]]:
    if not HAS_EMB:
        st.error("Embedding search requires sentence-transformers. Please install it.")
        return []
    if "passages_path" not in st.session_state or st.session_state["passages_path"] is None:
        st.error("Please select a passages.jsonl file in the sidebar.")
        return []
    pid2rec = st.session_state["pid2rec"]
    model = get_emb_model()
    qv = model.encode([query])
    texts = [pid2rec[pid]["text"] for pid in pid2rec]
    vecs = model.encode(texts)
    sims = cosine_similarity(qv, vecs)[0]
    pids = list(pid2rec.keys())
    order = np.argsort(-sims)
    out = [(pids[i], float(sims[i])) for i in order[:k]]
    return out

# -----------------------------
# Metrics: DCG/NDCG, Hit@k
# -----------------------------
def dcg(gains: List[int], k: int) -> float:
    gains_k = gains[:k]
    return sum(g / np.log2(i+2) for i, g in enumerate(gains_k))

def ndcg(gains: List[int], k: int) -> float:
    ideal = sorted(gains, reverse=True)
    denom = dcg(ideal, k)
    if denom == 0:
        return 0.0
    return dcg(gains, k) / denom

def hit_at_k(gains: List[int], k: int) -> float:
    return 1.0 if any(gains[:k]) else 0.0

def gains_from_pids(retrieved_pids: List[str], gold_pids: List[str], pid2rec: Dict[str, Dict], doc_level: bool) -> List[int]:
    if doc_level:
        gold_docs = set(pid_to_docid(pid, pid2rec) for pid in gold_pids)
        return [1 if pid_to_docid(pid, pid2rec) in gold_docs else 0 for pid in retrieved_pids]
    else:
        gold = set(gold_pids)
        return [1 if pid in gold else 0 for pid in retrieved_pids]

# -----------------------------
# LLM-as-judge scoring via Ollama
# -----------------------------
def ollama_score(prompt: str, model: str = "llama3.1:8b", key: Optional[str]=None) -> float:
    if not HAS_OLLAMA:
        st.error("Ollama python package not installed.")
        return 0.0
    try:
        # Local Ollama; ensure daemon is running: `ollama serve`
        resp = ollama.chat(model=model, messages=[{"role":"user","content": prompt}], options={"temperature": 0})
        text = resp["message"]["content"]
        # Expect a single float in [0,1]; robust parse:
        import re
        m = re.search(r"([01](?:\.\d+)?)", text)
        if not m:
            return 0.0
        return float(m.group(1))
    except Exception as e:
        st.warning(f"Ollama scoring failed: {e}")
        return 0.0

def judge_prompts():
    return {
        "context_relevance": (
            "You are grading how relevant the provided CONTEXT is to answering the QUESTION.\n"
            "Return a single number between 0 and 1 where 1 = perfectly relevant and 0 = irrelevant.\n"
            "QUESTION:\n{q}\n\nCONTEXT:\n{ctx}\n\nScore:"
        ),
        "answer_relevance": (
            "You are grading how well the ANSWER addresses the QUESTION, regardless of correctness.\n"
            "Return a number in [0,1]. 1 = fully addresses, 0 = not at all.\n"
            "QUESTION:\n{q}\n\nANSWER:\n{a}\n\nScore:"
        ),
        "groundedness": (
            "You are grading whether the ANSWER is fully supported by the CONTEXT passages.\n"
            "Return a number in [0,1]. 1 = every claim is supported (no hallucinations). 0 = unsupported.\n"
            "CONTEXT:\n{ctx}\n\nANSWER:\n{a}\n\nScore:"
        )
    }

def sentence_level_grounding(answer: str, ctx_text: str, model_name: str) -> pd.DataFrame:
    """
    Simple sentence-level support marking using an LLM judge.
    Returns a DataFrame with sentence and support score [0,1].
    """
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    rows = []
    for s in sents:
        p = "Rate how well this CLAIM is supported by the CONTEXT. Return a number in [0,1].\nCONTEXT:\n{}\n\nCLAIM:\n{}\n\nScore:".format(ctx_text, s)
        score = ollama_score(p, model=model_name) if HAS_OLLAMA else np.nan
        rows.append({"sentence": s, "support": score})
    return pd.DataFrame(rows)

# -----------------------------
# Answer generation via Ollama
# -----------------------------
def generate_answer_ollama(question: str, context: str, model: str = "llama3.1:8b", style: str = "concise") -> str:
    if not HAS_OLLAMA:
        return "[Ollama not installed]"
    sys_prompt = (
        "You are a helpful assistant. Use ONLY the provided context. "
        "Cite with bracketed indices like [1], [2] that refer to the chunk order. "
        "If the answer is not in the context, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nWrite a {style} answer with citations."
    try:
        resp = ollama.chat(model=model,
                           messages=[{"role":"system","content":sys_prompt},
                                     {"role":"user","content":user}],
                           options={"temperature": 0})
        return resp["message"]["content"]
    except Exception as e:
        return f"[Ollama generation failed: {e}]"

# -----------------------------
# UI
# -----------------------------
st.title("🔎 RAG Insight Console")
st.caption("Inspect retrieval, generate answers with Ollama, and grade context/answer relevance & groundedness.")

with st.sidebar:
    st.header("Data & Models")
    passages_path = st.text_input("Passages JSONL path", value="kb/chunks/passages.jsonl")
    dataset_path = st.text_input("Eval dataset (JSONL)", value="eval/datasets/questions_known.jsonl")
    method = st.selectbox("Retrieval method", ["Vector", "Hybrid", "Vector+RR", "Hybrid+RR"])
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    pool_k = st.number_input("Pool K (for RR)", min_value=20, max_value=1000, value=200, step=10)
    dedup_per_doc = st.checkbox("Deduplicate per doc (keep 1 chunk/doc)", value=True)
    doc_level_scoring = st.checkbox("Doc-level scoring for metrics", value=True)
    st.divider()
    st.subheader("Answering")
    use_ollama = st.checkbox("Use Ollama for answer", value=True)
    ollama_model = st.text_input("Ollama model", value="llama3.1:8b")
    style = st.selectbox("Answer style", ["concise", "detailed", "bullet points"])
    st.subheader("Judging")
    judge_mode = st.selectbox("Judge mode", ["Ollama LLM-as-judge", "Fast Embedding (cosine)"])
    if judge_mode == "Fast Embedding (cosine)":
        st.caption("Requires sentence-transformers. Scores in [0,1] via cosine.")
    st.divider()
    st.caption("Tip: place this app at project root so repo imports work.")

# Load passages
passages_file = Path(passages_path)
if passages_file.exists():
    try:
        pid2rec = load_passages(passages_file)
        st.session_state["pid2rec"] = pid2rec
        st.session_state["passages_path"] = passages_file
    except Exception as e:
        st.error(f"Failed to load passages: {e}")
else:
    st.warning("Passages file not found. Some features will be disabled.")

tabs = st.tabs(["Single Query", "Batch Eval"])

with tabs[0]:
    st.subheader("Single Query Inspector")
    q = st.text_area("Enter a question", height=80, placeholder="e.g., Do you accept walk-ins without referral?")
    run = st.button("Run")
    if run and q.strip():
        # Retrieve
        pid2rec = st.session_state.get("pid2rec", {})
        if not pid2rec:
            st.error("Passages not loaded.")
        else:
            if method in ("Vector", "Vector+RR"):
                initial = search_vector(q, k=int(max(top_k, pool_k)), pool_k=int(pool_k))
            else:
                initial = search_hybrid(q, k=int(max(top_k, pool_k)), pool_k=int(pool_k))
            init_pids = [pid for pid, _ in initial]
            # Optional dedup per doc
            if dedup_per_doc:
                seen = set()
                dedup = []
                for pid, sc in initial:
                    d = pid_to_docid(pid, pid2rec)
                    if d in seen:
                        continue
                    seen.add(d)
                    dedup.append((pid, sc))
                initial = dedup
                init_pids = [pid for pid, _ in initial]

            # Rerank if requested
            ranked = initial
            if method in ("Vector+RR", "Hybrid+RR"):
                ranked = rerank(q, init_pids, pid2rec, top_k=int(top_k))
            else:
                ranked = initial[:int(top_k)]

            # Build display
            rows = []
            for i, (pid, sc) in enumerate(ranked, start=1):
                rec = pid2rec.get(pid, {"text":"[missing]"})
                rows.append({
                    "rank": i,
                    "pid": pid,
                    "doc_id": pid_to_docid(pid, pid2rec),
                    "score": round(float(sc), 4),
                    "text": rec.get("text", "")[:400] + ("..." if len(rec.get("text",""))>400 else "")
                })
            df = pd.DataFrame(rows)
            st.write("**Top Contexts**")
            st.dataframe(df, use_container_width=True)

            # Per-query retrieval metrics (needs golds from dataset if present)
            try:
                ds_file = Path(dataset_path)
                if ds_file.exists():
                    samples = load_questions(ds_file)
                    gold = []
                    for s in samples:
                        if s.get("question","").strip() == q.strip():
                            gold = s.get("gold_pids", [])
                            break
                    if gold:
                        gains = gains_from_pids([pid for pid, _ in ranked], gold, pid2rec, doc_level_scoring)
                        st.info(f"NDCG@{top_k}: {ndcg(gains, int(top_k)):.3f} | Hit@{top_k}: {hit_at_k(gains, int(top_k)):.3f}")
                    else:
                        st.caption("No gold labels found for this question in the dataset; metrics skipped.")
                else:
                    st.caption("Dataset not found; retrieval metrics skipped.")
            except Exception as e:
                st.warning(f"Metric calc failed: {e}")

            # Compose context block with bracketed indices
            ctx_blocks = []
            for i, (pid, _) in enumerate(ranked, start=1):
                txt = pid2rec.get(pid, {}).get("text", "")
                ctx_blocks.append(f"[{i}] {txt}")
            ctx_text = "\n\n".join(ctx_blocks)

            # Generate answer
            answer = ""
            if use_ollama:
                answer = generate_answer_ollama(q, ctx_text, model=ollama_model, style=style)
            else:
                answer = "(Answer generation disabled)"
            st.markdown("### Answer")
            st.write(answer)

            # Judge: context relevance, answer relevance, groundedness
            prompts = judge_prompts()
            if judge_mode == "Ollama LLM-as-judge":
                ctx_rel = ollama_score(prompts["context_relevance"].format(q=q, ctx=ctx_text), model=ollama_model) if HAS_OLLAMA else np.nan
                ans_rel = ollama_score(prompts["answer_relevance"].format(q=q, a=answer), model=ollama_model) if HAS_OLLAMA else np.nan
                ground = ollama_score(prompts["groundedness"].format(ctx=ctx_text, a=answer), model=ollama_model) if HAS_OLLAMA else np.nan
            else:
                # Fast embedding mode
                if HAS_EMB:
                    model = get_emb_model()
                    qv = model.encode([q])
                    ctxv = model.encode([ctx_text])
                    anv = model.encode([answer])
                    ctx_rel = float(cosine_similarity(qv, ctxv)[0][0])
                    ans_rel = float(cosine_similarity(qv, anv)[0][0])
                    ground = float(cosine_similarity(ctxv, anv)[0][0])
                    # Normalize to [0,1] (cosine may be [-1,1])
                    def norm01(x): return (x+1)/2
                    ctx_rel, ans_rel, ground = map(norm01, (ctx_rel, ans_rel, ground))
                else:
                    ctx_rel = ans_rel = ground = np.nan

            col1, col2, col3 = st.columns(3)
            col1.metric("Context Relevance", f"{ctx_rel:.2f}" if ctx_rel==ctx_rel else "N/A")
            col2.metric("Answer Relevance", f"{ans_rel:.2f}" if ans_rel==ans_rel else "N/A")
            col3.metric("Groundedness", f"{ground:.2f}" if ground==ground else "N/A")

            # Optional: sentence-level grounding heatmap
            if judge_mode == "Ollama LLM-as-judge" and HAS_OLLAMA and answer.strip():
                st.markdown("#### Sentence-level Support")
                df_sup = sentence_level_grounding(answer, ctx_text, model_name=ollama_model)
                st.dataframe(df_sup, use_container_width=True)

with tabs[1]:
    st.subheader("Batch Evaluation (dataset)")
    if st.button("Run Batch Eval"):
        ds_file = Path(dataset_path)
        pid2rec = st.session_state.get("pid2rec", {})
        if not ds_file.exists():
            st.error("Dataset not found.")
        elif not pid2rec:
            st.error("Passages not loaded.")
        else:
            samples = load_questions(ds_file)
            rows = []
            for s in samples:
                q = s["question"]
                gold = s.get("gold_pids", [])
                # Retrieve
                if method in ("Vector", "Vector+RR"):
                    initial = search_vector(q, k=int(max(top_k, pool_k)), pool_k=int(pool_k))
                else:
                    initial = search_hybrid(q, k=int(max(top_k, pool_k)), pool_k=int(pool_k))
                init_pids = [pid for pid,_ in initial]

                # Dedup
                if dedup_per_doc:
                    seen = set(); dedup = []
                    for pid, sc in initial:
                        d = pid_to_docid(pid, pid2rec)
                        if d in seen: continue
                        seen.add(d); dedup.append((pid, sc))
                    initial = dedup
                    init_pids = [pid for pid,_ in initial]

                # RR
                if method in ("Vector+RR", "Hybrid+RR"):
                    ranked = rerank(q, init_pids, pid2rec, top_k=int(top_k))
                else:
                    ranked = initial[:int(top_k)]

                gains = gains_from_pids([pid for pid,_ in ranked], gold, pid2rec, doc_level_scoring)
                row = {
                    "question": q,
                    "NDCG@k": ndcg(gains, int(top_k)),
                    "Hit@k": hit_at_k(gains, int(top_k)),
                    "retrieved_pids": [pid for pid,_ in ranked],
                    "gold_pids": gold
                }
                # Judge (optional quick aggregate)
                ctx_blocks = []
                for i, (pid, _) in enumerate(ranked, start=1):
                    txt = pid2rec.get(pid, {}).get("text", "")
                    ctx_blocks.append(f"[{i}] {txt}")
                ctx_text = "\\n\\n".join(ctx_blocks)
                if use_ollama and judge_mode == "Ollama LLM-as-judge" and HAS_OLLAMA:
                    prompts = judge_prompts()
                    row["ContextRel"] = ollama_score(prompts["context_relevance"].format(q=q, ctx=ctx_text))
                    # Answer only if you want; can be slow:
                    # ans = generate_answer_ollama(q, ctx_text, model=ollama_model, style="concise")
                    # row["AnsRel"] = ollama_score(prompts["answer_relevance"].format(q=q, a=ans))
                    # row["Grounded"] = ollama_score(prompts["groundedness"].format(ctx=ctx_text, a=ans))
                rows.append(row)

            df = pd.DataFrame(rows)
            st.write("### Results")
            st.dataframe(df, use_container_width=True)
            st.write("**Macro Averages**")
            st.write(df[["NDCG@k","Hit@k"]].mean())

            # Offer CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "batch_eval_results.csv", "text/csv")
