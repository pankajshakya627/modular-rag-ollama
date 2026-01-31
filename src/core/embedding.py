"""Embedding wrapper using LangChain for Modular RAG."""
from typing import Any, Dict, List, Optional, Union
from langchain_ollama import OllamaEmbeddings
import numpy as np
import logging

from .config import get_embedding_config, EmbeddingConfig

logger = logging.getLogger(__name__)

# Export OllamaEmbedding as alias for backward compatibility
OllamaEmbedding = OllamaEmbeddings


class EmbeddingWrapper:
    """LangChain-based Embedding wrapper for Ollama."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding wrapper with LangChain."""
        self.config = config or get_embedding_config()
        
        # Create LangChain Ollama Embeddings
        self.embedding = OllamaEmbeddings(
            model=self.config.model,
            base_url=self.config.base_url,
        )
        
        logger.info(f"Initialized LangChain Ollama Embeddings: {self.config.model}")
    
    def embed(self, text: str) -> List[float]:
        """Embed a single text using LangChain."""
        try:
            result = self.embedding.embed_query(text)
            return result
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using LangChain."""
        try:
            results = self.embedding.embed_documents(texts)
            return results
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query (same as embed for Ollama)."""
        return self.embed(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (batch operation)."""
        return self.embed_batch(texts)
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions."""
        # Generate a test embedding to get dimensions
        test_embedding = self.embed("test")
        return len(test_embedding)
    
    def normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length."""
        normalized = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized.append([x / norm for x in emb])
            else:
                normalized.append(emb)
        return normalized
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    
    def compute_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """Compute the centroid of a set of embeddings."""
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        return centroid.tolist()
    
    def find_closest_embeddings(self, query_embedding: List[float], 
                                 embeddings: List[List[float]], 
                                 top_k: int = 5) -> List[int]:
        """Find the closest embeddings to a query embedding."""
        if not embeddings:
            return []
        
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k indices
        return [idx for idx, _ in similarities[:top_k]]
    
    def cluster_embeddings(self, embeddings: List[List[float]], 
                           num_clusters: int = 5) -> List[int]:
        """Cluster embeddings using K-means."""
        from sklearn.cluster import KMeans
        
        if len(embeddings) < num_clusters:
            num_clusters = len(embeddings)
        
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        return cluster_labels.tolist()


class EmbeddingWrapperWithCache(EmbeddingWrapper):
    """Embedding wrapper with caching support."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, cache_size: int = 1000):
        """Initialize with caching."""
        super().__init__(config)
        self.cache_size = cache_size
        self._cache = {}
    
    def embed(self, text: str) -> List[float]:
        """Embed with caching."""
        if text in self._cache:
            return self._cache[text]
        
        result = super().embed(text)
        
        # Add to cache
        if len(self._cache) < self.cache_size:
            self._cache[text] = result
        
        return result


# Singleton instance
_embedding_instance: Optional[EmbeddingWrapper] = None


def get_embedding() -> EmbeddingWrapper:
    """Get the global embedding instance."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingWrapper()
    return _embedding_instance
