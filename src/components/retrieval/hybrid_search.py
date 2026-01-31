"""Hybrid search implementation combining BM25 and dense retrieval."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import re

from rank_bm25 import BM25Okapi

from .document_processor import Document, DocumentChunk
from .vector_store import (
    BaseVectorStore,
    LangChainChromaVectorStore,
    VectorStoreManager,
    SearchResult,
)

logger = logging.getLogger(__name__)


class BM25Searcher:
    """BM25 sparse retrieval using token-based matching."""
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        top_k: int = 20,
    ):
        """Initialize the BM25 searcher."""
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization: lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def build_index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents."""
        self.documents = []
        self.doc_ids = []
        
        for doc in documents:
            for chunk in doc.chunks:
                self.documents.append(chunk.content)
                self.doc_ids.append(f"{doc.id}_{chunk.chunk_index}")
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(
            tokenized_docs,
            k1=self.k1,
            b=self.b,
        )
        
        logger.info(f"Built BM25 index with {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search using BM25."""
        if self.bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index first.")
        
        top_k = top_k or self.top_k
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # Normalize score to [0, 1] range
            normalized_score = 1 / (1 + score)
            
            result = SearchResult(
                id=self.doc_ids[idx],
                content=self.documents[idx],
                score=normalized_score,
                metadata={},
                chunk_index=int(self.doc_ids[idx].split("_")[-1]),
            )
            results.append(result)
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the BM25 index."""
        # Rebuild index with new documents
        all_docs = self.documents
        all_ids = self.doc_ids
        
        for doc in documents:
            for chunk in doc.chunks:
                all_docs.append(chunk.content)
                all_ids.append(f"{doc.id}_{chunk.chunk_index}")
        
        tokenized_docs = [self._tokenize(doc) for doc in all_docs]
        self.bm25_index = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        logger.info(f"Added {len(documents)} documents to BM25 index")
    
    def clear(self) -> None:
        """Clear the BM25 index."""
        self.bm25_index = None
        self.documents = []
        self.doc_ids = []


class HybridSearcher:
    """Hybrid search combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        bm25_searcher: Optional[BM25Searcher] = None,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        top_k: int = 50,
    ):
        """Initialize the hybrid searcher."""
        self.vsm = vector_store_manager
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.fusion_method = fusion_method
        self.alpha = alpha  # Weight for dense retrieval
        self.top_k = top_k
        
        # Track documents for BM25
        self._documents = []
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: float = 60.0,
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        # Create ranking maps
        dense_rank = {r.id: rank for rank, r in enumerate(dense_results)}
        sparse_rank = {r.id: rank for rank, r in enumerate(sparse_results)}
        
        # Get all unique document IDs
        all_ids = set(r.id for r in dense_results + sparse_results)
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_ids:
            rank_d = dense_rank.get(doc_id, len(dense_results) + 1)
            rank_s = sparse_rank.get(doc_id, len(sparse_results) + 1)
            
            rrf_d = 1.0 / (k + rank_d)
            rrf_s = 1.0 / (k + rank_s)
            
            rrf_scores[doc_id] = self.alpha * rrf_d + (1 - self.alpha) * rrf_s
        
        # Sort by combined score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create result list
        all_results = {r.id: r for r in dense_results + sparse_results}
        results = []
        for doc_id in sorted_ids[:self.top_k]:
            result = all_results[doc_id]
            result.score = rrf_scores[doc_id]
            results.append(result)
        
        return results
    
    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
    ) -> List[SearchResult]:
        """Combine results using weighted fusion."""
        # Normalize scores
        max_dense = max(r.score for r in dense_results) if dense_results else 1
        max_sparse = max(r.score for r in sparse_results) if sparse_results else 1
        
        # Create score maps
        dense_scores = {r.id: r.score / max_dense for r in dense_results}
        sparse_scores = {r.id: r.score / max_sparse for r in sparse_results}
        
        # Get all unique document IDs
        all_ids = set(r.id for r in dense_results + sparse_results)
        
        # Calculate combined scores
        combined_scores = {}
        for doc_id in all_ids:
            d_score = dense_scores.get(doc_id, 0)
            s_score = sparse_scores.get(doc_id, 0)
            combined_scores[doc_id] = self.alpha * d_score + (1 - self.alpha) * s_score
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Create result list
        all_results = {r.id: r for r in dense_results + sparse_results}
        results = []
        for doc_id in sorted_ids[:self.top_k]:
            result = all_results[doc_id]
            result.score = combined_scores[doc_id]
            results.append(result)
        
        return results
    
    def _score_normalized_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
    ) -> List[SearchResult]:
        """Combine results using normalized score fusion."""
        # Create maps
        dense_map = {r.id: r for r in dense_results}
        sparse_map = {r.id: r for r in sparse_results}
        
        # Get all unique document IDs
        all_ids = set(r.id for r in dense_results + sparse_results)
        
        # Get all scores for normalization
        all_dense_scores = [dense_map[doc_id].score for doc_id in all_ids if doc_id in dense_map]
        all_sparse_scores = [sparse_map[doc_id].score for doc_id in all_ids if doc_id in sparse_map]
        
        min_d, max_d = min(all_dense_scores), max(all_dense_scores)
        min_s, max_s = min(all_sparse_scores), max(all_sparse_scores)
        
        # Calculate combined scores
        combined_scores = {}
        for doc_id in all_ids:
            d_score = dense_map[doc_id].score if doc_id in dense_map else 0
            s_score = sparse_map[doc_id].score if doc_id in sparse_map else 0
            
            # Normalize to [0, 1]
            if max_d > min_d:
                d_norm = (d_score - min_d) / (max_d - min_d)
            else:
                d_norm = 0
            
            if max_s > min_s:
                s_norm = (s_score - min_s) / (max_s - min_s)
            else:
                s_norm = 0
            
            combined_scores[doc_id] = self.alpha * d_norm + (1 - self.alpha) * s_norm
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Create result list
        all_results = {r.id: r for r in dense_results + sparse_results}
        results = []
        for doc_id in sorted_ids[:self.top_k]:
            result = all_results[doc_id]
            result.score = combined_scores[doc_id]
            results.append(result)
        
        return results
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval."""
        top_k = top_k or self.top_k
        
        # Get dense results (increase top_k for fusion)
        dense_top_k = top_k * 2
        dense_results = self.vsm.search(query, dense_top_k, filters)
        
        # Get sparse results
        sparse_top_k = top_k * 2
        sparse_results = self.bm25_searcher.search(query, sparse_top_k)
        
        # Fuse results based on method
        if self.fusion_method == "rrf":
            results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.fusion_method == "weighted":
            results = self._weighted_fusion(dense_results, sparse_results)
        elif self.fusion_method == "score_normalized":
            results = self._score_normalized_fusion(dense_results, sparse_results)
        else:
            # Default to RRF
            results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Return top_k results
        return results[:top_k]
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """Build BM25 index from documents."""
        self._documents = documents
        self.bm25_searcher.build_index(documents)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both dense and sparse indexes."""
        self._documents.extend(documents)
        self.bm25_searcher.add_documents(documents)
    
    def clear(self) -> None:
        """Clear all indexes."""
        self.vsm.clear_index()
        self.bm25_searcher.clear()
        self._documents = []
