#!/usr/bin/env python3
"""
make_chunks.py — Markdown → passage chunker with PID compatibility.

- Anchors paths at repo root (no CWD issues).
- Light front-matter parse (e.g., "source_url: ...").
- Split by headings, then word-window chunk with overlap.
- Title boost so BM25/encoders latch onto topic terms.
- **PID format**: <doc_stem>_<span_start>  (matches gold_pids like 'accessibility_0')
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD_DIR = ROOT / "kb" / "markdown"
OUT_DIR = ROOT / "kb" / "chunks"
OUT_PATH = OUT_DIR / "passages.jsonl"

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)', flags=re.M)

def read_front_matter_fields(text: str) -> Dict[str, str]:
    meta = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if line.lstrip().startswith("#"):
            break
        m = re.match(r'^\s*([A-Za-z0-9_\-]+)\s*:\s*(.+?)\s*$', line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
        else:
            break
    return meta

def markdown_to_plain(text: str) -> str:
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)              # fenced code
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', text)               # images
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)            # [text](url) -> text
    text = re.sub(r'`([^`]*)`', r'\1', text)                        # inline code
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_by_headings(md: str) -> List[Tuple[str, str]]:
    parts, last_pos, last_heading = [], 0, ""
    for m in HEADING_RE.finditer(md):
        if m.start() > last_pos:
            parts.append((last_heading, md[last_pos:m.start()]))
        last_heading = m.group(2).strip()
        last_pos = m.end()
    if last_pos < len(md):
        parts.append((last_heading, md[last_pos:]))
    return parts or [("", md)]

def word_chunks(words, chunk: int, overlap: int):
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    step = max(1, chunk - max(0, overlap))
    n = len(words)
    if n == 0:
        return
    i = 0
    while i < n:
        j = min(n, i + chunk)
        yield (i, j, words[i:j])
        if j == n:
            break
        i += step

def build_passages(md_path: Path, chunk: int, overlap: int, title_boost: int) -> List[Dict]:
    raw = md_path.read_text(encoding="utf-8", errors="ignore")
    meta = read_front_matter_fields(raw)
    source_url = meta.get("source_url", "")

    m = re.search(r'^\s*#\s+(.+)$', raw, flags=re.M)
    title = (m.group(1).strip() if m else md_path.stem.replace("_", " ").title())

    sections = split_by_headings(raw)
    out: List[Dict] = []
    doc_id = md_path.stem

    for _, (heading, body_md) in enumerate(sections):
        plain = markdown_to_plain(body_md)
        if not plain:
            continue
        words = plain.split()
        for start, end, wslice in word_chunks(words, chunk=chunk, overlap=overlap):
            content = " ".join(wslice)
            prefix = (f"[TITLE] {title}. " * max(0, title_boost)).strip()
            heading_prefix = f"[HEADING] {heading}. " if heading else ""
            text = f"{prefix} {heading_prefix}{content}".strip()

            pid = f"{doc_id}_{start}"  # PID compatibility

            out.append({
                "pid": pid,
                "doc_id": doc_id,
                "title": title,
                "heading": heading,
                "span_start": start,
                "span_end": end,
                "source_url": source_url,
                "text": text
            })
    return out

def main():
    ap = argparse.ArgumentParser(description="Make JSONL passages from Markdown files (PID-compatible).")
    ap.add_argument("--md_dir", type=Path, default=DEFAULT_MD_DIR, help="Dir with .md files (default: kb/markdown)")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="Output JSONL (default: kb/chunks/passages.jsonl)")
    ap.add_argument("--chunk_words", type=int, default=600, help="Words per chunk (default: 600)")
    ap.add_argument("--overlap_words", type=int, default=80, help="Word overlap (default: 80)")
    ap.add_argument("--title_boost", type=int, default=2, help="Times to prepend [TITLE] (default: 2)")
    args = ap.parse_args()

    if not args.md_dir.exists():
        raise SystemExit(f"Markdown directory not found: {args.md_dir}")

    passages: List[Dict] = []
    for p in sorted(args.md_dir.glob("*.md")):
        passages.extend(build_passages(p, args.chunk_words, args.overlap_words, args.title_boost))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in passages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(passages)} passages to {args.out}")
    if passages:
        print("Sample PID:", passages[0]["pid"])

if __name__ == "__main__":
    main()
