"""Cross-Encoder reranker implementation."""
from typing import Any, Dict, List, Optional
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from .base import BaseReranker, RerankedResult
from ..retrieval.vector_store import SearchResult
from ...utils.temporal_utils import extract_temporal_entities, calculate_temporal_score

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder reranker for query-document relevance scoring."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """Initialize the Cross-Encoder reranker."""
        self.model_name = model_name
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
        """Load Cross-Encoder model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Get expected output type
            self.num_labels = self.model.config.num_labels
            self.is_regression = self.num_labels == 1
            
            logger.info(f"Loaded Cross-Encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading Cross-Encoder model: {e}")
            raise
    
    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        # Tokenize
        encoded = self.tokenizer(
            query,
            document,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get score
        with torch.no_grad():
            outputs = self.model(**encoded)
            
            if self.is_regression:
                # For regression models, output is a single logit
                score = outputs.logits.item()
                # Sigmoid for probability
                score = 1 / (1 + np.exp(-score))
            else:
                # For classification models, use softmax
                probs = torch.softmax(outputs.logits, dim=1)
                # Use probability of positive class
                score = probs[0, 1].item() if self.num_labels == 2 else probs[0, -1].item()
        
        return score
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank results using Cross-Encoder with temporal awareness."""
        if not results:
            return []
        
        top_k = top_k or len(results)
        
        # Extract temporal entities from query for temporal-aware reranking
        query_temporal_entities = []
        if self.enable_temporal_scoring:
            query_temporal_entities = extract_temporal_entities(query)
            if query_temporal_entities:
                logger.info(f"Query temporal context: {[e.normalized for e in query_temporal_entities]}")
        
        reranked_results = []
        
        for result in results:
            # Score the pair
            score = self._score_pair(query, result.content)
            
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
        """Rerank multiple result sets in batch for efficiency."""
        if not queries or not results_per_query:
            return []
        
        all_reranked = []
        
        for query, results in zip(queries, results_per_query):
            reranked = self.rerank(query, results, top_k)
            all_reranked.append(reranked)
        
        return all_reranked
    
    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score multiple documents against a query."""
        scores = []
        
        for doc in documents:
            score = self._score_pair(query, doc)
            scores.append(score)
        
        return scores
    
    def compute_cross_scores(
        self,
        query: str,
        documents: List[str],
    ) -> Dict[str, float]:
        """Compute detailed cross-encoder scores."""
        scores = self.score_batch(query, documents)
        
        return {
            doc: score for doc, score in zip(documents, scores)
        }
