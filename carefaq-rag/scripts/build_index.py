from pathlib import Path
import os, json, numpy as np
from sentence_transformers import SentenceTransformer

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]         # .../carefaq-rag/
PASSAGES_PATH = ROOT / "kb" / "chunks" / "passages.jsonl"
INDEX_DIR     = ROOT / "kb" / "index"



def main():
    rows = [json.loads(l) for l in PASSAGES.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    pids  = [r["pid"] for r in rows]
    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    np.save(OUT_DIR/"embeddings.npy", embs.astype("float32"))
    (OUT_DIR/"pids.json").write_text(json.dumps(pids, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import faiss
        index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized dot
        index.add(embs.astype("float32"))
        faiss.write_index(index, str(OUT_DIR/"index.faiss"))
        print(f"Wrote FAISS index with {index.ntotal} vectors")
    except Exception as e:
        print("FAISS not available; embeddings saved for sklearn fallback")
    print("OK")

if __name__ == "__main__":
    main()