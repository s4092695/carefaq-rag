import re

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def citations_supported(ans_text, cited_urls, by_url) -> bool:
    a = _norm_text(ans_text)
    if not a: return False
    for url in cited_urls:
        for p in by_url.get(url, []):
            if a and _norm_text(p.get("text","")).find(a) != -1:
                return True
    return False

def gold_supported(gold_text, cited_urls, by_url) -> bool:
    g = _norm_text(gold_text)
    if not g: return False
    for url in cited_urls:
        for p in by_url.get(url, []):
            if _norm_text(p.get("text","")).find(g) != -1:
                return True
    return False
