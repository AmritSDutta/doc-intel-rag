import logging
from pathlib import Path
import pdfplumber
import json
from services.factory import load_config
from llama_index.core.node_parser import SentenceSplitter


# LlamaIndex splitter import local to avoid heavy import unless used
def chunk_text_llama(text, chunk_size=600, chunk_overlap=150):
    logging.info(f'chunking text: {text[:10] if text else None} . . ,chunk_size: {chunk_size},overlap: {chunk_overlap}')

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(text)
    return [getattr(n, "get_content", lambda: str(n))() for n in nodes]


def extract_text(path: Path) -> str:
    logging.info(f'extracting text from: {path}')
    txt = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt += (p.extract_text() or "") + "\n"
    return txt


def main():
    cfg = load_config()
    src = Path(cfg["ingest"]["sources"][0]["path"])
    out = Path("data/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = extract_text(src)
    chunks = chunk_text_llama(text, cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"])
    with out.open("w", encoding="utf-8") as fh:
        for i, c in enumerate(chunks):
            fh.write(json.dumps({"id": f"chunk-{i}", "text": c, "source": str(src)}) + "\n")
    print(f"wrote {len(chunks)} chunks -> {out}")


if __name__ == "__main__":
    main()
