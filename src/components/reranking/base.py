"""Base reranker class with Pydantic models for production-grade type safety."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

from ..retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class RerankedResult(BaseModel):
    """Represents a reranked search result with full Pydantic validation."""
    id: str
    content: str
    original_score: float = Field(ge=0.0, le=1.0)
    reranked_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: Optional[str] = None
    chunk_index: int = Field(default=0, ge=0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


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

        reranked_results = []
        for i in range(0, len(results), self.batch_size):
            batch = results[i : i + self.batch_size]

            for result in batch:
                # Support both Pydantic SearchResult and plain dict
                if isinstance(result, dict):
                    content = result.get("content", "")
                    r_id = result.get("id", "")
                    orig_score = result.get("score", 0.0)
                    metadata = result.get("metadata", {})
                    document_id = result.get("document_id")
                    chunk_index = result.get("chunk_index", 0)
                else:
                    content = result.content
                    r_id = result.id
                    orig_score = result.score
                    metadata = result.metadata
                    document_id = result.document_id
                    chunk_index = result.chunk_index

                score = self._score_result(query, content)
                reranked = RerankedResult(
                    id=r_id,
                    content=content,
                    original_score=min(1.0, max(0.0, orig_score)),
                    reranked_score=score,
                    metadata=metadata,
                    document_id=document_id,
                    chunk_index=chunk_index,
                )
                reranked_results.append(reranked)

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
        """Score a single result using LLM via invoke() (not generate())."""
        from langchain_core.messages import HumanMessage

        prompt_text = (
            f"You are a relevance scorer. Given a query and a document passage, "
            f"rate how relevant the passage is to answering the query on a scale of 0 to 1.\n\n"
            f"Query: {query}\n\n"
            f"Document Passage: {content[:500]}\n\n"
            f"Relevance Score (just a number between 0 and 1):"
        )

        try:
            # Use invoke() — the correct ChatOllama method
            response = self.llm_wrapper.invoke([HumanMessage(content=prompt_text)])
            response_text = response.content if hasattr(response, "content") else str(response)

            import re
            score_match = re.search(r"(\d+\.?\d*)", response_text)
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

        filtered = [r for r in reranked if r.reranked_score >= self.threshold]

        if len(filtered) < self.min_results:
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
    """Reranker that promotes diversity in results using MMR."""

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
        """Rerank results considering diversity via MMR."""
        reranked = self.base_reranker.rerank(query, results, top_k)

        if len(reranked) <= 1:
            return reranked

        contents = [r.content for r in reranked]
        embeddings = self.embedding_wrapper.embed_documents(contents)

        try:
            from ...core.embedding import normalize_embeddings
            embeddings = normalize_embeddings(embeddings)
        except ImportError:
            pass

        selected: List[RerankedResult] = []
        selected_embeddings: List[List[float]] = []

        import numpy as np

        for result, embedding in zip(reranked, embeddings):
            if not selected:
                selected.append(result)
                selected_embeddings.append(embedding)
                continue

            relevance = result.reranked_score
            emb_arr = np.array(embedding)
            max_similarity = max(
                float(np.dot(emb_arr, np.array(se)) /
                      (np.linalg.norm(emb_arr) * np.linalg.norm(se) + 1e-9))
                for se in selected_embeddings
            )

            mmr_score = (1 - self.diversity_weight) * relevance - self.diversity_weight * max_similarity

            if mmr_score > 0 or len(selected) < (top_k or len(reranked)):
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
