import os
from pathlib import Path

import chromadb
import pdfplumber
from google import genai
from llama_index.core.node_parser import SentenceSplitter


# use the same embedding helper you provided
def get_embedding_model(chunks):
    client = genai.Client()
    return client.models.embed_content(
        model="text-embedding-004",
        contents=chunks,
        config={
            "task_type": "semantic_similarity",
            "output_dimensionality": 256,
        },
    )


# absolute path to asset (test file lives in tests/)
PDF_PATH = Path(__file__).parent.joinpath("assets", "eval_source_document.pdf").resolve()


# --- use LlamaIndex SentenceSplitter for token-aware chunking ---
def llamaindex_chunk(text: str, chunk_size: int = 600, chunk_overlap: int = 150):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.split_text(text)
    # nodes are objects with get_content(); fall back to str() if necessary
    return [getattr(n, "get_content", lambda: str(n))() for n in nodes]


def test_llamaindex_chunking_plus_chroma_search():
    assert os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}"

    # extract text
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or "") + "\n"
    assert text.strip()

    # 1) produce LlamaIndex chunks
    li_chunks = llamaindex_chunk(text, chunk_size=500, chunk_overlap=100)
    assert li_chunks and all(isinstance(c, str) for c in li_chunks)

    # 2) embed via your genai helper and normalize the SDK objects simply
    raw_embs = get_embedding_model(li_chunks)
    normalized_raw_embeddings = raw_embs.embeddings
    embeddings = [e.values for e in normalized_raw_embeddings]  # simple inline normalization
    assert len(embeddings) == len(li_chunks)

    # 3) push to in-memory Chroma and query
    client = chromadb.Client()
    col = client.create_collection("li_eval_collection")
    ids = [f"li-{i}" for i in range(len(li_chunks))]
    metas = [{"source": os.path.basename(PDF_PATH), "li_chunk_index": i} for i in range(len(li_chunks))]

    col.add(ids=ids, documents=li_chunks, metadatas=metas, embeddings=embeddings)

    # 4) semantic query
    query = "What is this document used to evaluate?"
    q_raw = get_embedding_model([query])
    q_raw_normalized = q_raw.embeddings[0]
    q_emb = q_raw_normalized.values

    res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents", "metadatas", "distances"])
    assert res and res.get("documents"), "No documents returned"

    # sanity assertions about retrieval
    docs = res["documents"][0]
    dists = res["distances"][0]
    assert len(docs) == len(dists) and len(docs) <= 3
    assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))

    # cleanup: delete collection (avoid reset permission issues)
    client.delete_collection("li_eval_collection")
