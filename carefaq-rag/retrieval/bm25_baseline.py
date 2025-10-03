import json, re
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "kb" / "chunks" / "passages.jsonl"

passages = [json.loads(l) for l in PASSAGES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
docs = [p["text"] for p in passages]
tok  = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tok)

_aliases = [
    (re.compile(r"\breferral(s)?\b", flags=re.I), " referral refer referal "),
    (re.compile(r"\baccessibility\b", flags=re.I), " accessibility accessible access "),
]

def _expand(q: str) -> str:
    out = q
    for pat, add in _aliases:
        if pat.search(out):
            out += " " + add
    return out

def search(q, k=5):
    qx = _expand(q).lower().split()
    scores = bm25.get_scores(qx)
    idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    out = []
    for i in idxs:
        p = passages[i]
        out.append({
            "pid": p["pid"],
            "source_url": p.get("source_url", ""),
            "text": p["text"],
            "score": float(scores[i]),
            "method": "bm25",
        })
    return out
