from typing import Sequence, List


class EmbeddingService:
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError
