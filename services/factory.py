import logging

import yaml
from pathlib import Path

from services.embedding.genai_service import GenAIEmbeddingService
from services.llm.base import LLMService
from services.llm.genai_llm_service import GenAILLMService
from services.vectorstores.chroma_store import ChromaStore


def load_config(path="config/settings.yaml"):
    with open(path, "r") as fh:
        logging.info('Loading config')
        return yaml.safe_load(fh)


def get_embedding_service(cfg=None):
    cfg = cfg or load_config()
    if cfg["embeddings"]["provider"] == "google.genai":
        logging.info('Creating GenAIEmbeddingService')
        return GenAIEmbeddingService(api_key=cfg.get("google_api_key"))
    raise RuntimeError("Unknown embedding provider")


def get_vector_store(cfg=None):
    cfg = cfg or load_config()
    if cfg["vector_store"]["type"] == "chroma":
        logging.info('Creating chromadb store')
        return ChromaStore(collection_name=cfg["vector_store"].get("collection_name", "doc_intel_eval"))
    raise RuntimeError("Unknown vectorstore")


def get_llm_service(cfg=None) -> LLMService:
    return GenAILLMService()
