import re, math, statistics
from typing import List, Tuple

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def token_f1(pred: str, gold: str) -> float:
    ps = normalize_text(pred).split()
    gs = normalize_text(gold).split()
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    common = {}
    for t in gs:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ps:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    prec = overlap / len(ps) if ps else 0.0
    rec  = overlap / len(gs) if gs else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def ndcg_at_k(gains: List[float], k: int) -> float:
    gains = gains[:k]
    dcg = sum((g / math.log2(i+2)) for i, g in enumerate(gains))
    idcg = sum((g / math.log2(i+2)) for i, g in enumerate(sorted(gains, reverse=True)))
    return 0.0 if idcg == 0 else dcg / idcg

def mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0