"""Cross-encoder reranking using LangChain CrossEncoderReranker.

Replaces custom CrossEncoderReranker with:
- langchain_community.cross_encoders.HuggingFaceCrossEncoder
- langchain.retrievers.document_compressors.CrossEncoderReranker
- langchain.retrievers.ContextualCompressionRetriever

The temporal scoring logic is preserved as a post-processing step (no LangChain equivalent).
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangChainDocument

# LangChain cross-encoder reranking
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker as LCCrossEncoderReranker
    from langchain_classic.retrievers import ContextualCompressionRetriever
    LANGCHAIN_RERANKER_AVAILABLE = True
except ImportError:
    LANGCHAIN_RERANKER_AVAILABLE = False

# Temporal scoring (custom — no LangChain equivalent)
from ...utils.temporal_utils import extract_temporal_entities, calculate_temporal_score

from .base import BaseReranker, RerankedResult

logger = logging.getLogger(__name__)


def create_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n: int = 10,
) -> Optional['LCCrossEncoderReranker']:
    """Create a LangChain CrossEncoderReranker.
    
    Args:
        model_name: HuggingFace cross-encoder model name
        top_n: Number of top documents to return
        
    Returns:
        Configured CrossEncoderReranker or None if not available
    """
    if not LANGCHAIN_RERANKER_AVAILABLE:
        logger.warning("langchain cross-encoder reranker not available. "
                       "Install: pip install langchain-community sentence-transformers")
        return None
    
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = LCCrossEncoderReranker(model=model, top_n=top_n)
    
    logger.info(f"Created CrossEncoderReranker with model: {model_name}, top_n={top_n}")
    return compressor


def create_compression_retriever(
    base_retriever,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n: int = 10,
) -> Optional['ContextualCompressionRetriever']:
    """Create a ContextualCompressionRetriever with cross-encoder reranking.
    
    This wraps a base retriever with cross-encoder reranking, so that
    retriever.invoke(query) automatically reranks the results.
    
    Args:
        base_retriever: The underlying retriever to wrap
        model_name: HuggingFace cross-encoder model name
        top_n: Number of top documents to return
        
    Returns:
        ContextualCompressionRetriever or None if not available
    """
    compressor = create_cross_encoder_reranker(model_name, top_n)
    if compressor is None:
        return None
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    
    logger.info("Created ContextualCompressionRetriever with CrossEncoder reranking")
    return compression_retriever


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using LangChain's built-in CrossEncoderReranker.
    
    This class provides the same interface as the original custom reranker
    but delegates to LangChain's implementation. Temporal scoring (custom)
    is applied as a post-processing step.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_n: int = 10,
        temporal_boost: float = 1.5,
        temporal_penalty: float = 0.7,
    ):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
            top_n: Number of top documents to return
            temporal_boost: Score multiplier for temporal matches
            temporal_penalty: Score multiplier for temporal mismatches
        """
        self.model_name = model_name
        self.top_n = top_n
        self.temporal_boost = temporal_boost
        self.temporal_penalty = temporal_penalty
        
        # Initialize LangChain cross-encoder
        self._lc_reranker = None
        self._hf_model = None
        self._init_reranker()
    
    def _init_reranker(self):
        """Initialize the LangChain cross-encoder components."""
        if LANGCHAIN_RERANKER_AVAILABLE:
            try:
                self._hf_model = HuggingFaceCrossEncoder(model_name=self.model_name)
                self._lc_reranker = LCCrossEncoderReranker(
                    model=self._hf_model,
                    top_n=self.top_n,
                )
                logger.info(f"Initialized LangChain CrossEncoderReranker: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoderReranker: {e}")
                self._lc_reranker = None
        else:
            logger.warning("LangChain cross-encoder reranker not available")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank results using LangChain cross-encoder + temporal scoring.
        
        Args:
            query: The original query
            results: List of search results (dicts with 'content', 'id', 'score', etc.)
            top_k: Number of top results to return
        """
        top_k = top_k or self.top_n
        
        if not results:
            return []
        
        # Convert to LangChain Documents for the reranker
        lc_docs = [
            LangChainDocument(
                page_content=r.get("content", ""),
                metadata={
                    "id": r.get("id", ""),
                    "original_score": r.get("score", 0.0),
                    **r.get("metadata", {}),
                },
            )
            for r in results
        ]
        
        # Use LangChain cross-encoder for reranking
        if self._lc_reranker is not None:
            try:
                reranked_docs = self._lc_reranker.compress_documents(lc_docs, query)
            except Exception as e:
                logger.error(f"CrossEncoder reranking failed: {e}")
                reranked_docs = lc_docs
        else:
            reranked_docs = lc_docs
        
        # Apply temporal scoring (custom, no LangChain equivalent)
        query_temporal = extract_temporal_entities(query)
        
        reranked_results = []
        for idx, doc in enumerate(reranked_docs[:top_k]):
            # Get the relevance score from cross-encoder
            ce_score = doc.metadata.get("relevance_score", 1.0 / (idx + 1))
            original_score = doc.metadata.get("original_score", 0.0)
            
            # Apply temporal scoring
            doc_temporal = extract_temporal_entities(doc.page_content)
            temporal_multiplier = calculate_temporal_score(
                query_temporal,
                doc_temporal,
                match_boost=self.temporal_boost,
                mismatch_penalty=self.temporal_penalty,
            )
            
            final_score = ce_score * temporal_multiplier
            
            reranked_results.append(RerankedResult(
                id=doc.metadata.get("id", ""),
                content=doc.page_content,
                original_score=original_score,
                reranked_score=final_score,
                metadata={
                    **doc.metadata,
                    "cross_encoder_score": ce_score,
                    "temporal_multiplier": temporal_multiplier,
                },
            ))
        
        # Sort by final reranked score 
        reranked_results.sort(key=lambda r: r.reranked_score, reverse=True)
        
        return reranked_results[:top_k]
