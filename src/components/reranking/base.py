"""Base reranker class."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from ..retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Represents a reranked search result."""
    id: str
    content: str
    original_score: float
    reranked_score: float
    metadata: Dict[str, Any] = None
    document_id: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "original_score": self.original_score,
            "reranked_score": self.reranked_score,
            "metadata": self.metadata or {},
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank search results."""
        pass
    
    @abstractmethod
    def rerank_batch(
        self,
        queries: List[str],
        results_per_query: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankedResult]]:
        """Rerank multiple result sets."""
        pass


class LLMEnsembleReranker(BaseReranker):
    """Reranker using LLM ensemble for relevance scoring."""
    
    def __init__(self, llm_wrapper, batch_size: int = 5):
        """Initialize the LLM ensemble reranker."""
        self.llm_wrapper = llm_wrapper
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank results using LLM-based scoring."""
        if not results:
            return []
        
        top_k = top_k or len(results)
        
        # Score results in batches
        reranked_results = []
        for i in range(0, len(results), self.batch_size):
            batch = results[i:i + self.batch_size]
            
            for result in batch:
                score = self._score_result(query, result.content)
                reranked = RerankedResult(
                    id=result.id,
                    content=result.content,
                    original_score=result.score,
                    reranked_score=score,
                    metadata=result.metadata,
                    document_id=result.document_id,
                    chunk_index=result.chunk_index,
                )
                reranked_results.append(reranked)
        
        # Sort by reranked score
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        return reranked_results[:top_k]
    
    def rerank_batch(
        self,
        queries: List[str],
        results_per_query: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankedResult]]:
        """Rerank multiple result sets."""
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_per_query)
        ]
    
    def _score_result(self, query: str, content: str) -> float:
        """Score a single result using LLM."""
        prompt = f"""You are a relevance scorer. Given a query and a document passage, 
rate how relevant the passage is to answering the query on a scale of 0 to 1.

Query: {query}

Document Passage: {content[:500]}

Relevance Score (0-1):"""
        
        try:
            response = self.llm_wrapper.llm.generate(prompt, max_tokens=10)
            
            # Extract score from response
            import re
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.error(f"Error scoring result: {e}")
            return 0.5


class ThresholdReranker(BaseReranker):
    """Reranker that filters results by a relevance threshold."""
    
    def __init__(
        self,
        base_reranker: BaseReranker,
        threshold: float = 0.5,
        min_results: int = 1,
    ):
        """Initialize the threshold reranker."""
        self.base_reranker = base_reranker
        self.threshold = threshold
        self.min_results = min_results
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank and filter results by threshold."""
        reranked = self.base_reranker.rerank(query, results, top_k)
        
        # Filter by threshold
        filtered = [r for r in reranked if r.reranked_score >= self.threshold]
        
        # Ensure minimum results
        if len(filtered) < self.min_results:
            # Add top results from reranked list
            for r in reranked:
                if r not in filtered:
                    filtered.append(r)
                    if len(filtered) >= self.min_results:
                        break
        
        return filtered
    
    def rerank_batch(
        self,
        queries: List[str],
        results_per_query: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankedResult]]:
        """Rerank and filter multiple result sets."""
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_per_query)
        ]


class DiversityReranker(BaseReranker):
    """Reranker that promotes diversity in results."""
    
    def __init__(
        self,
        base_reranker: BaseReranker,
        embedding_wrapper,
        diversity_weight: float = 0.3,
    ):
        """Initialize the diversity reranker."""
        self.base_reranker = base_reranker
        self.embedding_wrapper = embedding_wrapper
        self.diversity_weight = diversity_weight
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank results considering diversity."""
        # First get relevance scores
        reranked = self.base_reranker.rerank(query, results, top_k)
        
        if len(reranked) <= 1:
            return reranked
        
        # Calculate diversity
        # Get embeddings for all results
        contents = [r.content for r in reranked]
        embeddings = self.embedding_wrapper.embed_texts(contents, normalize=True)
        
        # Select diverse results using MMR (Maximal Marginal Relevance)
        selected = []
        selected_embeddings = []
        
        for result, embedding in zip(reranked, embeddings):
            if not selected:
                selected.append(result)
                selected_embeddings.append(embedding)
                continue
            
            # Calculate MMR score
            relevance = result.reranked_score
            max_similarity = 0.0
            
            for sel_emb in selected_embeddings:
                similarity = self.embedding_wrapper.embedding.compute_similarity(
                    embedding, sel_emb
                )
                max_similarity = max(max_similarity, similarity)
            
            mmr_score = (1 - self.diversity_weight) * relevance - self.diversity_weight * max_similarity
            
            # Add to selected if score is high enough
            if mmr_score > 0 or len(selected) < top_k:
                selected.append(result)
                selected_embeddings.append(embedding)
        
        return selected[:top_k] if top_k else selected
    
    def rerank_batch(
        self,
        queries: List[str],
        results_per_query: List[List[SearchResult]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankedResult]]:
        """Rerank multiple result sets with diversity."""
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_per_query)
        ]
