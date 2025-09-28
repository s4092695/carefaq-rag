import streamlit as st
from gen.refusal import needs_refusal, REFUSAL_TEXT
import subprocess, json

st.title("CareFAQ — Medical Centre (Admin FAQs only)")
q = st.text_input("Ask a question")

def bm25(q):
    out = subprocess.check_output(["python", "retrieval/bm25_baseline.py", q]).decode().strip().splitlines()
    return [eval(l) for l in out] if out and out[0] else []


from gen.rag_chain import answer as rag_answer

if st.button("Ask") and q:
    out = rag_answer(q, k=5)
    st.write(out["answer"])
    if out["sources"]:
        st.write("**Sources:**")
        for u in out["sources"]:
            st.markdown(f"- [{u}]({u})")
    st.caption(f"Method: {out.get('method','-')}")
