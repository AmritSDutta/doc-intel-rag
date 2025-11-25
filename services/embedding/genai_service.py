import logging
from typing import Sequence, List
from google import genai
from .base import EmbeddingService


class GenAIEmbeddingService(EmbeddingService):
    """google genai embedding implementation as EmbeddingService."""
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model: str = "text-embedding-004"
        self.dimension: int = 256
        self.type: str = "semantic_similarity"

    def embed(self, texts: Sequence[str],
              task_type: str = None,
              output_dimensionality: int = None) -> List[List[float]]:
        """google genai embedding for text collections."""
        logging.info(f'embedding, task_type: {task_type}, output_dimensionality: {output_dimensionality}')
        resp = self.client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config={
                "task_type": task_type if task_type else self.type,
                "output_dimensionality": output_dimensionality if output_dimensionality else self.dimension,
            },
        )
        return [e.values for e in resp.embeddings]

    def embed_batch(
            self,
            texts: Sequence[str],
            batch_size: int = 32,
            task_type: str = None,
            output_dimensionality: int = None,
    ):
        """Batch embedding for large text collections."""

        texts = list(texts)
        if not texts:
            return []

        final_embeddings = []

        # determine config
        cfg = {
            "task_type": task_type if task_type else self.type,
            "output_dimensionality": output_dimensionality if output_dimensionality else self.dimension,
        }

        # main batching loop
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            resp = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=cfg,
            )

            # normalize each returned embedding
            final_embeddings.extend([e.values for e in resp.embeddings])

        return final_embeddings

