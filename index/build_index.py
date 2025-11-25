import json
from pathlib import Path
from services.factory import get_embedding_service, get_vector_store

CHUNKS_FILE = Path("data/chunks.jsonl")


def main(chunk_file: Path):
    embedder = get_embedding_service()
    store = get_vector_store()
    rows = [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines()]
    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    metas = [{"source": row["source"], "i": i} for i, row in enumerate(rows)]
    embs = embedder.embed(texts)
    store.save(ids, texts, metas, embs)
    print(f"indexed {len(ids)} chunks")


if __name__ == "__main__":
    main()
