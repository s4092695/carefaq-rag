# CareFAQ RAG — Medical Centre

## Quick start
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/make_chunks.py
python retrieval/bm25_baseline.py "Do you bulk bill pensioners?"
streamlit run app.py
pytest -q
