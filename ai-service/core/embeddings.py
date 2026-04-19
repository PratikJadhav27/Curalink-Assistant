"""
Embedding Service - Singleton wrapper around sentence-transformers
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    _instance: SentenceTransformer = None
    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        if cls._instance is None:
            cls._instance = SentenceTransformer(cls.MODEL_NAME)
        return cls._instance

    @classmethod
    def embed(cls, texts: list[str]) -> np.ndarray:
        model = cls.get_instance()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    @classmethod
    def cosine_similarity(cls, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity (vectors are already L2-normalized)."""
        return np.dot(doc_vecs, query_vec)
