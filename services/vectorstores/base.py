from typing import Sequence, Dict, List


class VectorStore:
    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        raise NotImplementedError

    def query(self, query_embedding: Sequence[float], n_results: int = 3) -> Dict:
        raise NotImplementedError

    def delete_collection(self, name: str): ...
