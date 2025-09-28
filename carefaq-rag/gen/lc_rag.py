# gen/lc_rag.py
from typing import Dict, List
import re, time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from gen.lc_llm import make_llm
from retrieval.get_contexts import get_contexts, CONFIG

def _format_docs(ctxs: List[Dict]) -> str:
    # Keep short chunks; include SOURCE line for each so the model can cite
    blocks = []
    for i, c in enumerate(ctxs, 1):
        txt = re.sub(r"\s+", " ", c.get("text","")).strip()
        blocks.append(f"[{i}] {txt}\nSOURCE: {c.get('source_url','')}")
    return "\n\n".join(blocks)

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant for clinic FAQs. "
     "Answer ONLY with facts from <CONTEXT>. If the answer is not clearly in the context, reply exactly with NO_ANSWER. "
     "Prefer copying short sentences verbatim. Add bracketed citations like [1] or [2] matching the context blocks you used."),
    ("human", "Question: {question}\n\n<CONTEXT>\n{context}\n</CONTEXT>")
])

def _to_sources(ctxs: List[Dict], answer_text: str) -> List[str]:
    # if the model cited [k], keep only those URLs; else return top-1 URL
    import re
    cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text))
    if cited:
        urls = []
        for i, c in enumerate(ctxs, 1):
            if i in cited and c.get("source_url"):
                urls.append(c["source_url"])
        return sorted(set(urls))
    # fallback: the highest-scoring context
    for c in ctxs:
        if c.get("source_url"):
            return [c["source_url"]]
    return []

def answer_lc(question: str, k: int | None = None) -> Dict:
    t0 = time.time()
    k = k or CONFIG.get("top_k", 5)
    ctxs = get_contexts(question, k=k)

    # Low-confidence guardrail like your extractive path
    scores = [c.get("score", 0.0) for c in ctxs]
    if not scores or max(scores) < 0.20:
        return {"answer": "", "sources": [], "method": "noanswer", "latency_ms": round((time.time()-t0)*1000,1)}

    llm = make_llm()
    chain = (
        RunnableParallel(
            question=lambda x: x["question"],
            source_documents=lambda x: x["source_documents"]
        )
        | RunnableLambda(lambda d: {"question": d["question"], "context": _format_docs(d["source_documents"])})
        | PROMPT
        | llm
        | StrOutputParser()
    )

    text = chain.invoke({"question": question, "source_documents": ctxs}).strip()

    if text == "NO_ANSWER":
        return {"answer": "", "sources": [], "method": "noanswer", "latency_ms": round((time.time()-t0)*1000,1)}

    # Keep it concise: take first ~2 sentences (helps your eval faithfulness)
    short = re.split(r'(?<=[.!?])\s+', text.strip())
    short_text = " ".join(short[:2]).strip()

    sources = _to_sources(ctxs, short_text)
    return {
        "answer": short_text,
        "sources": sources,
        "method": "gen-lc",
        "latency_ms": round((time.time()-t0)*1000,1)
    }
