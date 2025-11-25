import os
from pathlib import Path

import pytest
import pdfplumber
import chromadb
from google import genai

PDF_PATH = Path(__file__).parent.joinpath("assets", "eval_source_document.pdf").resolve()


def chunk_text(s: str, chunk_size: int = 800, overlap: int = 200):
    i, chunks = 0, []
    while i < len(s):
        chunks.append(s[i:i + chunk_size].strip())
        i += chunk_size - overlap
    return [c for c in chunks if c]


def get_embedding_model(chunks):
    client = genai.Client()
    return client.models.embed_content(
        model="text-embedding-004",
        contents=chunks,
        config={
            "task_type": 'semantic_similarity',
            "output_dimensionality": 256
        }
    ).embeddings


@pytest.mark.slow
def test_ingest_embed_chroma_roundtrip():
    # preconditions

    assert os.path.exists(PDF_PATH), f"PDF not found at {PDF_PATH}"

    # 1) extract text from the provided PDF
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or "") + "\n"
    assert text.strip(), "Extracted text is empty."

    # 2) chunk
    chunks = chunk_text(text, chunk_size=600, overlap=150)
    assert chunks, "No chunks produced."

    # 3) embed using Google GenAI client
    emb_resp = get_embedding_model(chunks)
    embeddings_normalized = [e.values for e in emb_resp]
    assert len(embeddings_normalized) == len(chunks)

    # 4) insert into in-memory Chroma
    chroma_client = chromadb.Client()
    col = chroma_client.create_collection("eval_docs_genai_pytest")
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metas = [{"source": os.path.basename(PDF_PATH), "chunk_index": i} for i in range(len(chunks))]
    col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings_normalized)

    # 5) query (semantic)
    query = "What is the document used to evaluate?"
    q_emb = get_embedding_model([query])
    q_emb_normalized = q_emb[0].values
    # client.models.embed_content(model="text-embedding-004", contents=[query]).embeddings[0]

    res = col.query(
        query_embeddings=q_emb_normalized,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    assert res and res.get("documents"), "No results returned from Chroma query."

    dists = res["distances"][0]
    assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1)), "Returned distances not ordered."

    # cleanup
    chroma_client.delete_collection("eval_docs_genai_pytest")
