# gen/rag_chain.py
import re
from typing import Dict, List
from retrieval.get_contexts import get_contexts
from gen.unanswerable import needs_refusal, low_confidence, REFUSAL_TEXT

SENT = re.compile(r"(?<=[.!?])\s+")

def _pick_sentence(text: str, query: str) -> str:
    # very simple extractive heuristic: first sentence containing a query keyword, else first sentence
    toks = [w for w in re.findall(r"[A-Za-z0-9%$']+", query.lower()) if len(w) > 2]
    sents = re.split(SENT, text.strip())
    for s in sents:
        ls = s.lower()
        if any(t in ls for t in toks):
            return s.strip()
    return sents[0].strip() if sents else text.strip()

def answer(query: str, k: int = 5) -> Dict:
    if needs_refusal(query):
        return {"answer": REFUSAL_TEXT, "sources": [], "method": "refusal"}

    ctx = get_contexts(query, k)
    scores = [c.get("score", 0.0) for c in ctx]
    if low_confidence(scores):
        # echo safe no-answer
        return {"answer": "Sorry, I couldn’t find this in our clinic information.", "sources": [], "method": "noanswer"}

    # Compose short extractive answer and attach citations
    top = ctx[0] if ctx else {}
    snippet = _pick_sentence(top.get("text", ""), query)
    sources = list(dict.fromkeys([c.get("source_url","") for c in ctx if c.get("source_url")]))[:3]
    return {
        "answer": snippet if snippet else "Please check the linked source.",
        "sources": sources,
        "method": top.get("method","vector"),
        "debug": {"pids": [c.get("pid") for c in ctx], "scores": scores}
    }
