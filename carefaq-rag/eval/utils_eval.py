# eval/utils_eval.py
import json, re
from pathlib import Path

def project_root() -> Path:
    # eval/utils_eval.py -> eval/ -> carefaq-rag/
    return Path(__file__).resolve().parents[1]

def load_passages(passages_path: str | None = None):
    if passages_path is None:
        passages_path = project_root() / "kb/chunks/passages.jsonl"
    else:
        passages_path = Path(passages_path)

    rows = [json.loads(l) for l in passages_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_pid = {r["pid"]: r for r in rows}
    by_url = {}
    for r in rows:
        url = (r.get("source_url") or "").strip()
        if url:
            by_url.setdefault(url, []).append(r)
    return rows, by_pid, by_url
