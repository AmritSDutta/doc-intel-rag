from typing import Sequence, Dict
import chromadb
from .base import VectorStore


class ChromaStore(VectorStore):
    def __init__(self, collection_name="doc_intel_eval"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        try:
            self.col = self.client.get_collection(self.collection_name)
        except Exception:
            self.col = self.client.create_collection(self.collection_name)

    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        self.col.add(ids=list(ids), documents=list(docs), metadatas=list(metas), embeddings=list(embeddings))

    def query(self, query_embedding: Sequence[float], n_results: int = 3) -> Dict:
        return self.col.query(query_embeddings=[list(query_embedding)], n_results=n_results,
                              include=["documents", "metadatas", "distances"])

    def delete_collection(self, name: str):
        self.client.delete_collection(name)
