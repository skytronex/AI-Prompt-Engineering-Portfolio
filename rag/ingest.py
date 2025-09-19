# rag/ingest.py
import os, json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOCS = BASE / "sample_docs.md"
OUT = BASE / "vector_store.json"

def dummy_embedding(text, dim=32):
    v = [0.0]*dim
    for i,ch in enumerate(text[:dim]):
        v[i] = (ord(ch) % 97)/97.0
    return v

def build():
    if not DOCS.exists():
        print("No sample docs found at", DOCS)
        return 1
    docs = []
    with open(DOCS, "r", encoding="utf-8") as f:
        blocks = f.read().split("\n\n")
    for i,b in enumerate(blocks):
        text = b.strip()
        if not text:
            continue
        emb = dummy_embedding(text)
        docs.append({"id": f"doc_{i}", "text": text, "embedding": emb, "source": str(DOCS)})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)
    print(f"Wrote {len(docs)} docs to {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(build())
