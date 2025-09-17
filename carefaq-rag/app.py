import streamlit as st
from gen.refusal import needs_refusal, REFUSAL_TEXT
import subprocess, json

st.title("CareFAQ — Medical Centre (Admin FAQs only)")
q = st.text_input("Ask a question")

def bm25(q):
    out = subprocess.check_output(["python", "retrieval/bm25_baseline.py", q]).decode().strip().splitlines()
    return [eval(l) for l in out] if out and out[0] else []

if st.button("Ask") and q:
    if needs_refusal(q):
        st.warning(REFUSAL_TEXT)
    else:
        ctx = bm25(q)
        if not ctx:
            st.write("Sorry, I couldn't find this in our info.")
        else:
            st.write("**Answer (baseline placeholder):** Check the linked source for details.")
            st.write("**Sources:**")
            for c in ctx:
                st.write(f"- {c.get('source_url','')}")
