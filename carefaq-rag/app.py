# app.py
import os, json, re
import streamlit as st
from retrieval.get_contexts import get_contexts

# --- Config ---
METHODS = {
    "Vector": "vector",
    "Vector + Rerank": "vector_rr",
    "Hybrid": "hybrid",
    "Hybrid + Rerank": "hybrid_rr",
}
K_DEFAULT = 5

# Optional: simple out-of-scope guardrail for your domain
DOMAIN_HINTS = [
    "canberra health services", "walk-in centre", "chs", "referral",
    "accessibility", "telehealth", "fees", "hours", "vaccination"
]
def is_in_scope(q: str) -> bool:
    ql = q.lower()
    return any(tok in ql for tok in DOMAIN_HINTS)

# Optional: generator (Ollama)
def generate_answer(question: str, contexts: list[str]) -> str:
    if not os.getenv("USE_OLLAMA"):
        # no generator configured; retrieval-only demo is still valid
        bullet = "\n".join(f"- {c[:300]}..." for c in contexts)
        return f"(no generator configured) Top evidence:\n{bullet}"
    import requests
    prompt = (
        "You are a CHS Walk-in Centre assistant. Answer ONLY from the provided context. "
        "If the answer is not clearly in the context, say 'I couldn't find that in our info.'\n\n"
        f"Question: {question}\n\n"
        "Context:\n" + "\n\n".join(contexts) + "\n\nAnswer:"
    )
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": os.getenv("OLLAMA_MODEL","llama3.1:8b"), "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json().get("response","")

# --- UI ---
st.set_page_config(page_title="CareFAQ RAG Demo", layout="centered")
st.title("CareFAQ RAG Demo")
q = st.text_input("Ask a question about Canberra Health Services / Walk-in Centres:")
method_label = st.selectbox("Retrieval method", list(METHODS.keys()), index=1)
k = st.slider("k (contexts)", 1, 10, K_DEFAULT)

if st.button("Search") and q.strip():
    if not is_in_scope(q):
        st.warning("This looks out of scope for our CHS domain. Please ask about Walk-in Centres, referrals, hours, fees, accessibility, etc.")
    with st.spinner("Retrieving..."):
        hits = get_contexts(q, k=k, method=METHODS[method_label])
    if not hits:
        st.error("No contexts found.")
    else:
        # Show contexts with simple term highlighting
        st.subheader("Top contexts")
        pat = re.compile(re.escape(q.split("?")[0].split()[0]), re.I) if q.split() else None
        ctx_texts = []
        for i, h in enumerate(hits, 1):
            txt = h.get("text","")
            if pat: txt = pat.sub(lambda m: f"**{m.group(0)}**", txt)
            st.markdown(f"**{i}. {h['pid']}**  \nScore: `{h.get('score',0):.3f}`  \n{txt}")
            ctx_texts.append(h.get("text",""))
        # Optional LLM answer (or retrieval-only summary)
        st.subheader("Answer")
        ans = generate_answer(q, ctx_texts[:k])
        st.markdown(ans)
