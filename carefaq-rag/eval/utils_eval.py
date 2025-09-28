import json
from pathlib import Path

def load_passages(passages_path = "carefaq-rag/kb/chunks/passages.jsonl"):
    path = Path(passages_path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_pid = {r["pid"]: r for r in rows}
    by_url = {}
    for r in rows:
        by_url.setdefault(r.get("source_url",""), []).append(r)
    return rows, by_pid, by_url