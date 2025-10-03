# scripts/build_index.py
import os, json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "kb" / "chunks" / "passages.jsonl"
INDEX_DIR     = ROOT / "kb" / "index"

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    lines = PASSAGES_PATH.read_text(encoding="utf-8").splitlines()
    passages = [json.loads(l) for l in lines if l.strip()]
    texts = [p["text"] for p in passages]
    pids  = [p["pid"]  for p in passages]

    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    np.save(INDEX_DIR / "embeddings.npy", embs)
    (INDEX_DIR / "pids.json").write_text(json.dumps(pids, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import faiss
        index = faiss.IndexFlatIP(embs.shape[1])  # normalized vectors → cosine via dot
        index.add(embs)
        faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
        print(f"Wrote FAISS index with {index.ntotal} vectors")
    except Exception as e:
        print("FAISS not available; saved embeddings/pids only:", e)
    print("OK")

if __name__ == "__main__":
    main()
