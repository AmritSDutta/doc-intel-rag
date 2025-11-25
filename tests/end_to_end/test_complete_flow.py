import json
import tempfile
from pathlib import Path

from config.logging_config import setup_logging
from ingest.ingest_pdfs import extract_text, chunk_text_llama
from query.run_query import search_and_synthesize
from services.factory import load_config, get_embedding_service, get_vector_store


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent  # root/


def test_complete_flow():
    setup_logging()

    ROOT: Path = _project_root()
    CONFIG_PATH = ROOT / "config" / "settings.yaml"
    cfg = load_config(CONFIG_PATH)

    src = ROOT / 'tests' / Path(cfg["ingest"]["sources"][0]["path"])
    text = extract_text(src)
    chunks = chunk_text_llama(text, cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"])
    out = Path(tempfile.gettempdir()) / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(chunks):
            record = {
                "id": f"chunk-{i}",
                "text": c,
                "source": str(src),
            }
            fh.write(json.dumps(record) + "\n")
    print(f"chunking done, wrote {len(chunks)} chunks -> {out}")

    embedder = get_embedding_service(cfg)
    store = get_vector_store(cfg)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    metas = [{"source": row["source"], "i": i} for i, row in enumerate(rows)]
    embs = embedder.embed(texts)
    store.save(ids, texts, metas, embs)
    print(f"indexed {len(ids)} chunks")

    answer, hits = search_and_synthesize('what are the key concepts?', cfg=cfg)
    print(answer)
