$proj = "carefaq-rag"
$today = Get-Date -Format "yyyy-MM-dd"
if (Test-Path $proj) { Write-Error "Directory exists. Stop."; exit 1 }

New-Item -ItemType Directory -Force -Path $proj,"$proj/docs","$proj/kb/markdown","$proj/kb/chunks","$proj/retrieval","$proj/gen","$proj/eval","$proj/tests","$proj/results","$proj/scripts","$proj/.github/workflows" | Out-Null

@"
.venv/
__pycache__/
*.pyc
.env
.streamlit/
results/*.tmp
"@ | Set-Content "$proj/.gitignore"

@"
streamlit
rank-bm25
scikit-learn
pytest
"@ | Set-Content "$proj/requirements.txt"

@"
# CareFAQ RAG — Medical Centre

## Quick start
`python -m venv .venv; .\.venv\Scripts\Activate.ps1`
`pip install -r requirements.txt`
`python scripts/make_chunks.py`
`python retrieval/bm25_baseline.py "Do you bulk bill pensioners?"`
`streamlit run app.py`
`pytest -q`
"@ | Set-Content "$proj/README.md"

@"
# PRD (v0)
Aim: Build a test-driven RAG assistant for a medical centre that answers administrative FAQs with citations and safely refuses clinical advice.
In-scope: hours, fees & bulk-billing, telehealth, repeat scripts, referrals, results policy, vaccinations, accessibility/parking, after-hours.
Out-of-scope: diagnosis/treatment advice/dosing; emergencies -> "Call 000".
"@ | Set-Content "$proj/docs/PRD_final.md"

@"
Refusal text:
"I can’t provide medical advice. For urgent symptoms call 000 or visit an emergency department. For non-urgent issues, please book with a GP."
"@ | Set-Content "$proj/docs/Safety_and_Refusal_final.md"

@"
Risks: non-responsive team; over-refusal; KB drift.
Decisions: Chunk=600, Overlap=80; Top-k=5; Start with BM25.
"@ | Set-Content "$proj/docs/Risks_Decisions.md"

@"
title,url,last_updated,section
Hours & Contact,https://exampleclinic.com/hours,$today,General
Fees & Billing,https://exampleclinic.com/fees,$today,Billing
Telehealth,https://exampleclinic.com/telehealth,$today,Appointments
"@ | Set-Content "$proj/kb/manifest.csv"

@"
Chunk size = 600 tokens, overlap = 80.
"@ | Set-Content "$proj/kb/CHUNKING.md"

@"
source_url: https://exampleclinic.com/fees
page_section: Fees & billing
last_updated: $today

Our clinic fees vary by appointment type. We bulk bill eligible pensioners and children under 16 on weekdays. Please bring your Medicare card. After-hours fees differ.
"@ | Set-Content "$proj/kb/markdown/fees.md"

@"
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
"@ | Set-Content "$proj/scripts/make_chunks.py"

@"
import json, sys
from rank_bm25 import BM25Okapi
passages = [json.loads(l) for l in open("kb/chunks/passages.jsonl", encoding="utf-8")]
docs = [p["text"] for p in passages]
tok = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tok)
def search(q, k=5):
    scores = bm25.get_scores(q.lower().split())
    idx = range(len(scores))
    top = sorted(zip(idx, scores), key=lambda x: -x[1])[:k]
    return [{"pid": passages[i]["pid"], "source_url": passages[i]["source_url"], "score": float(s)} for i,s in top]
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Do you bulk bill pensioners?"
    for r in search(q, k=5):
        print(r)
"@ | Set-Content "$proj/retrieval/bm25_baseline.py"

@"
EMERGENCY = ["chest pain","severe","bleeding","unconscious","difficulty breathing"]
CLINICAL  = ["dosage","diagnose","prescribe","side effect","should i take","treat"]
REFUSAL_TEXT = ("I can’t provide medical advice. For urgent symptoms call 000 or visit an emergency department. "
                "For non-urgent issues, please book with a GP.")
def needs_refusal(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in EMERGENCY+CLINICAL)
"@ | Set-Content "$proj/gen/refusal.py"

@"
import streamlit as st
from gen.refusal import needs_refusal, REFUSAL_TEXT
import subprocess, json

st.title("CareFAQ — Medical Centre (Admin FAQs only)")
q = st.text_input("Ask a question")

def bm25(q):
    out = subprocess.check_output(["python", "retrieval/bm25_baseline.py", q]).decode().strip().splitlines()
    return [eval(l) for l in out] if out and out[0] else []

if st.button("Ask") and q:
    if needs_refusal(q):
        st.warning(REFUSAL_TEXT)
    else:
        ctx = bm25(q)
        if not ctx:
            st.write("Sorry, I couldn't find this in our info.")
        else:
            st.write("**Answer (baseline placeholder):** Check the linked source for details.")
            st.write("**Sources:**")
            for c in ctx:
                st.write(f"- {c.get('source_url','')}")
"@ | Set-Content "$proj/app.py"

@"
{"qid":"q001","type":"known","question":"Do you bulk bill pensioners?","gold_pages":["fees.md"]}
{"qid":"q002","type":"out_of_scope","question":"I have chest pain—what should I do?"}
"@ | Set-Content "$proj/eval/questions_v1.jsonl"

@"
def test_sanity():
    assert True
"@ | Set-Content "$proj/tests/test_sanity.py"

@"
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m venv .venv
      - run: .\.venv\Scripts\Activate.ps1; pip install -U pip; pip install -r requirements.txt
      - run: .\.venv\Scripts\Activate.ps1; python scripts/make_chunks.py
      - run: .\.venv\Scripts\Activate.ps1; pytest -q
"@ | Set-Content "$proj/.github/workflows/ci.yml"

Write-Host "✅ Scaffolding complete in .\$proj"
Write-Host "Next steps:"
Write-Host "  cd $proj"
Write-Host "  python -m venv .venv; .\.venv\Scripts\Activate.ps1"
Write-Host "  pip install -r requirements.txt"
Write-Host "  python scripts/make_chunks.py"
Write-Host "  python retrieval/bm25_baseline.py 'Do you bulk bill pensioners?'"
Write-Host "  streamlit run app.py"
Write-Host "  pytest -q"
