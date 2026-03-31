"""Embedding factory using LangChain OllamaEmbeddings for Modular RAG.

Production improvements:
- Thread-safe singleton via functools.lru_cache (no mutable module globals)
- Async-ready: OllamaEmbeddings supports aembed_query / aembed_documents
- batch_size enforcement from config
"""
import functools
import logging
from typing import List, Optional

from langchain_ollama import OllamaEmbeddings
import numpy as np

from .config import get_embedding_config, EmbeddingConfig

logger = logging.getLogger(__name__)


def _make_embedding(config: EmbeddingConfig) -> OllamaEmbeddings:
    """Internal factory — creates a configured OllamaEmbeddings instance.

    OllamaEmbeddings supports:
    - .embed_query(text)           → List[float]
    - .aembed_query(text)          → List[float]   (async)
    - .embed_documents(texts)      → List[List[float]]
    - .aembed_documents(texts)     → List[List[float]] (async)
    """
    embedding = OllamaEmbeddings(
        model=config.model,
        base_url=config.base_url,
    )
    logger.info(
        f"Initialized OllamaEmbeddings: model={config.model}, "
        f"base_url={config.base_url}, batch_size={config.batch_size}"
    )
    return embedding


@functools.lru_cache(maxsize=4)
def _cached_embedding(model: str, base_url: str) -> OllamaEmbeddings:
    """Thread-safe cached embedding factory keyed on (model, base_url).

    lru_cache is thread-safe in CPython — guarantees one instance per config.
    """
    from .config import EmbeddingConfig
    cfg = EmbeddingConfig(model=model, base_url=base_url)
    return _make_embedding(cfg)


def get_embedding(config: Optional[EmbeddingConfig] = None) -> OllamaEmbeddings:
    """Get a thread-safe cached OllamaEmbeddings instance.

    Uses lru_cache keyed on (model, base_url) — avoids mutable module-level singletons.
    """
    cfg = config or get_embedding_config()
    return _cached_embedding(model=cfg.model, base_url=cfg.base_url)


def create_embedding(config: Optional[EmbeddingConfig] = None) -> OllamaEmbeddings:
    """Create a new (non-cached) OllamaEmbeddings instance."""
    cfg = config or get_embedding_config()
    return _make_embedding(cfg)


def embed_in_batches(
    texts: List[str],
    embedding: OllamaEmbeddings,
    batch_size: int = 32,
) -> List[List[float]]:
    """Embed texts in configurable batches, respecting Ollama's limits.

    Args:
        texts: List of texts to embed
        embedding: OllamaEmbeddings instance
        batch_size: Number of texts per batch (from EmbeddingConfig.batch_size)

    Returns:
        List of embedding vectors
    """
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = embedding.embed_documents(batch)
        all_embeddings.extend(batch_embs)
        logger.debug(f"Embedded batch {i // batch_size + 1}: {len(batch)} texts")
    return all_embeddings


async def aembed_in_batches(
    texts: List[str],
    embedding: OllamaEmbeddings,
    batch_size: int = 32,
) -> List[List[float]]:
    """Async version of embed_in_batches."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = await embedding.aembed_documents(batch)
        all_embeddings.extend(batch_embs)
    return all_embeddings


# ============================================================================
# Embedding math utilities
# ============================================================================

def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Normalize embeddings to unit length."""
    normalized = []
    for emb in embeddings:
        arr = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        normalized.append((arr / norm).tolist() if norm > 0 else emb)
    return normalized


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    emb1 = np.array(embedding1, dtype=np.float32)
    emb2 = np.array(embedding2, dtype=np.float32)
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
    return np.mean(np.array(embeddings, dtype=np.float32), axis=0).tolist()


def cluster_embeddings(embeddings: List[List[float]], num_clusters: int = 5) -> List[int]:
    """Cluster embeddings using K-means."""
    from sklearn.cluster import KMeans

    n = len(embeddings)
    if n == 0:
        return []
    num_clusters = min(num_clusters, n)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.array(embeddings, dtype=np.float32))
    return labels.tolist()


# Backward compatibility aliases
EmbeddingWrapper = OllamaEmbeddings
OllamaEmbedding = OllamaEmbeddings
