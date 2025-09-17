from pathlib import Path
import json, re
CHUNK=600; OVERLAP=80
out = []
for p in Path("kb/markdown").glob("*.md"):
    raw = p.read_text(encoding="utf-8")
    url = re.search(r"^source_url:\s*(.*)$", raw, flags=re.M).group(1).strip()
    body = "\n".join(raw.splitlines()[3:])
    words = body.split()
    for i in range(0, max(1,len(words)), CHUNK-OVERLAP):
        chunk = " ".join(words[i:i+CHUNK])
        out.append({"pid": f"{p.stem}_{i}", "source_url": url, "text": chunk})
Path("kb/chunks").mkdir(parents=True, exist_ok=True)
Path("kb/chunks/passages.jsonl").write_text("\n".join(json.dumps(x) for x in out), encoding="utf-8")
print(f"wrote {len(out)} passages")
