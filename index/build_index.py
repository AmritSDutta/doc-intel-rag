import json
from pathlib import Path
from services.factory import get_embedding_service, get_vector_store

CHUNKS_FILE = Path("data/chunks.jsonl")


def main():
    embedder = get_embedding_service()
    store = get_vector_store()
    rows = [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]
    texts = [r["text"] for r in rows]
    ids = [r["id"] for r in rows]
    metas = [{"source": r["source"], "i": i} for i, r in enumerate(rows)]
    embs = embedder.embed(texts)
    store.save(ids, texts, metas, embs)
    print(f"indexed {len(ids)} chunks")


if __name__ == "__main__":
    main()
