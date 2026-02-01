"""ColBERT (Late Interaction) reranker implementation."""
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from .base import BaseReranker, RerankedResult
from ..retrieval.vector_store import SearchResult
from ...utils.temporal_utils import extract_temporal_entities, calculate_temporal_score

logger = logging.getLogger(__name__)


class ColBERTReranker(BaseReranker):
    """ColBERT reranker using late interaction for token-level matching."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """Initialize the ColBERT reranker."""
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        
        # Temporal scoring configuration
        self.temporal_boost = 1.5  # Boost for matching dates
        self.temporal_penalty = 0.7  # Penalty for mismatched dates
        self.enable_temporal_scoring = True
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load ColBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded ColBERT model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading ColBERT model: {e}")
            raise
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text into token embeddings."""
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use last hidden state
            embeddings = outputs.last_hidden_state
        
        return embeddings  # Shape: (1, seq_len, hidden_dim)
    
    def _maxsim(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> float:
        """Calculate MaxSim score between query and document.
        
        For each query token, find the maximum similarity with any document token,
        then sum across all query tokens.
        """
        # Normalize embeddings
        query_norm = query_embeddings / (
            query_embeddings.norm(dim=-1, keepdim=True) + 1e-8
        )
        doc_norm = doc_embeddings / (
            doc_embeddings.norm(dim=-1, keepdim=True) + 1e-8
        )
        
        # Compute similarity matrix: (query_len, doc_len)
        similarity = torch.matmul(query_norm.squeeze(0), doc_norm.squeeze(0).T)
        
        # MaxSim: max similarity for each query token
        max_sim, _ = similarity.max(dim=1)
        
        # Sum across query tokens
        score = max_sim.sum().item()
        
        # Normalize by query length
        score = score / query_embeddings.size(1)
        
        return score
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank results using ColBERT late interaction with temporal awareness."""
        if not results:
            return []
        
        top_k = top_k or len(results)
        
        # Encode query once
        query_embedding = self._encode_text(query)
        
        # Extract temporal entities from query for temporal-aware reranking
        query_temporal_entities = []
        if self.enable_temporal_scoring:
            query_temporal_entities = extract_temporal_entities(query)
            if query_temporal_entities:
                logger.info(f"Query temporal context: {[e.normalized for e in query_temporal_entities]}")
        
        reranked_results = []
        
        for result in results:
            # Encode document
            doc_embedding = self._encode_text(result.content)
            
            # Calculate MaxSim score
            score = self._maxsim(query_embedding, doc_embedding)
            
            # Apply temporal scoring if enabled and query has temporal entities
            if self.enable_temporal_scoring and query_temporal_entities:
                doc_temporal_entities = extract_temporal_entities(result.content)
                temporal_multiplier = calculate_temporal_score(
                    query_temporal_entities,
                    doc_temporal_entities,
                    match_boost=self.temporal_boost,
                    mismatch_penalty=self.temporal_penalty,
                )
                score = score * temporal_multiplier
            
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
    
    def compute_similarity_matrix(
        self,
        query: str,
        documents: List[str],
    ) -> np.ndarray:
        """Compute similarity matrix between query and all documents."""
        # Encode query
        query_embedding = self._encode_text(query)
        
        # Compute scores for all documents
        scores = []
        for doc in documents:
            doc_embedding = self._encode_text(doc)
            score = self._maxsim(query_embedding, doc_embedding)
            scores.append(score)
        
        return np.array(scores)
