# gen/rag_chain.py
import re, time
from typing import Dict, List
from retrieval.get_contexts import get_contexts, CONFIG
from gen.unanswerable import needs_refusal, low_confidence
import os
USE_LANGCHAIN = os.getenv("USE_LANGCHAIN_RAG", "0") == "1"
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')  # split on sentence end or newlines
_TIME_PAT = re.compile(r'\b(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{1,2}(:\d{2})?\s?(am|pm)\b', re.I)
_FEES_PAT = re.compile(r'\b(bulk[-\s]?bill|medicare|free|no\s+cost|fees?)\b', re.I)

def _split_sentences(text: str) -> List[str]:
    # collapse spaces, keep punctuation
    text = re.sub(r'\s+', ' ', (text or '')).strip()
    # break bullets into pseudo-sentences
    text = text.replace('•', '. ').replace('- ', '')
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    # de-dupe, cap length
    out, seen = [], set()
    for s in sents:
        s_norm = s.lower()
        if s_norm in seen: 
            continue
        seen.add(s_norm)
        # trim very long sentences
        tokens = s.split()
        if len(tokens) > 45:
            s = ' '.join(tokens[:45]).rstrip(',;') + '...'
        out.append(s)
    return out

def _score_sentence(q: str, s: str) -> float:
    ql = set(re.findall(r'[a-z0-9]+', q.lower()))
    sl = set(re.findall(r'[a-z0-9]+', s.lower()))
    # basic overlap
    overlap = len(ql & sl) / max(1, len(ql))
    bonus = 0.0
    if _TIME_PAT.search(s): bonus += 0.25     # hours-like signals
    if _FEES_PAT.search(s): bonus += 0.25     # fees/bulk-billing signals
    if len(s) < 220:        bonus += 0.10     # prefer concise
    return overlap + bonus

def _pick_best_sentence(query: str, contexts: List[Dict]) -> Dict:
    # build candidate sentences from each context
    candidates = []
    for c in contexts:
        sents = _split_sentences(c.get("text",""))[:8]   # first few sentences are usually the most on-topic
        for s in sents:
            candidates.append({
                "sent": s,
                "score": _score_sentence(query, s),
                "url": c.get("source_url",""),
                "pid": c.get("pid",""),
                "base_score": float(c.get("score", 0.0)),
                "method": c.get("method","vector")
            })
    if not candidates:
        return {}
    best = max(candidates, key=lambda x: (x["score"], x["base_score"]))
    # (optionally) join top-2 if they’re short and from the same source
    companions = [c for c in candidates if c["url"] == best["url"]]
    companions = sorted(companions, key=lambda x: -x["score"])
    answer_text = best["sent"]
    for c in companions[1:2]:
        if len(answer_text) < 160 and c["sent"] not in answer_text:
            answer_text = f"{answer_text} {c['sent']}"
            break
    return {
        "answer": answer_text.strip(),
        "sources": [u for u in [best["url"]] if u],
        "method": best["method"]
    }

def answer(query: str, k: int = None) -> Dict:
    t0 = time.time()
    if needs_refusal(query):
        return {
            "answer": "I can’t provide medical advice. For urgent symptoms call 000 or visit an emergency department. For non-urgent issues, please book with a GP.",
            "sources": [],
            "method": "refusal",
            "latency_ms": round((time.time()-t0)*1000, 1)
        }

    k = k or CONFIG.get("top_k", 5)
    contexts = get_contexts(query, k=k)
    scores = [c.get("score", 0.0) for c in contexts]
    if low_confidence(scores, threshold=0.20):
        return {"answer": "", "sources": [], "method": "noanswer", "latency_ms": round((time.time()-t0)*1000, 1)}
    from gen.lc_rag import answer_lc
    if USE_LANGCHAIN:
        return answer_lc(query, k=k)
    picked = _pick_best_sentence(query, contexts)
    if not picked.get("answer"):
        return {"answer": "", "sources": [], "method": "noanswer", "latency_ms": round((time.time()-t0)*1000, 1)}

    picked["latency_ms"] = round((time.time()-t0)*1000, 1)
    return picked
