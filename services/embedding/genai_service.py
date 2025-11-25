from typing import Sequence, List
from google import genai
from .base import EmbeddingService


class GenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = "text-embedding-004"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self.client.models.embed_content(model=self.model, contents=list(texts))
        return [e.values for e in resp.embeddings]
