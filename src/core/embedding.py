"""Embedding factory using LangChain OllamaEmbeddings for Modular RAG.

Instead of a custom wrapper, this module provides a factory function
that returns a properly configured OllamaEmbeddings instance.
All embedding functionality (embed_query, embed_documents, etc.)
is provided directly by the LangChain OllamaEmbeddings class.

Additional math utilities (similarity, clustering) are provided
as standalone functions.
"""
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
import numpy as np
import logging

from .config import get_embedding_config, EmbeddingConfig

logger = logging.getLogger(__name__)


def create_embedding(config: Optional[EmbeddingConfig] = None) -> OllamaEmbeddings:
    """Create a configured OllamaEmbeddings instance.
    
    Returns a LangChain OllamaEmbeddings that supports:
    - .embed_query(text) -> List[float]
    - .embed_documents(texts) -> List[List[float]]
    """
    config = config or get_embedding_config()
    
    embedding = OllamaEmbeddings(
        model=config.model,
        base_url=config.base_url,
    )
    
    logger.info(f"Initialized OllamaEmbeddings: {config.model}")
    return embedding


# Singleton instance
_embedding_instance: Optional[OllamaEmbeddings] = None


def get_embedding() -> OllamaEmbeddings:
    """Get the global OllamaEmbeddings instance."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = create_embedding()
    return _embedding_instance


# ============================================================================
# Embedding math utilities (no LangChain equivalent, kept as standalone)
# ============================================================================

def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize embeddings to unit length."""
    normalized = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized.append([x / norm for x in emb])
        else:
            normalized.append(emb)
    return normalized


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 > 0 and norm2 > 0:
        return float(dot_product / (norm1 * norm2))
    return 0.0


def compute_centroid(embeddings: List[List[float]]) -> List[float]:
    """Compute the centroid of a set of embeddings."""
    if not embeddings:
        return []
    embeddings_array = np.array(embeddings)
    centroid = np.mean(embeddings_array, axis=0)
    return centroid.tolist()


def cluster_embeddings(embeddings: List[List[float]], num_clusters: int = 5) -> List[int]:
    """Cluster embeddings using K-means."""
    from sklearn.cluster import KMeans
    
    if len(embeddings) < num_clusters:
        num_clusters = len(embeddings)
    
    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    return cluster_labels.tolist()


# Backward compatibility aliases
EmbeddingWrapper = OllamaEmbeddings  # Type alias for imports that reference EmbeddingWrapper
OllamaEmbedding = OllamaEmbeddings   # Legacy alias
