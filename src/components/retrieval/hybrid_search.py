"""Hybrid search using LangChain BM25Retriever and EnsembleRetriever.

Replaces custom BM25Searcher and HybridSearcher with:
- langchain_community.retrievers.BM25Retriever (for sparse/keyword search)
- langchain.retrievers.EnsembleRetriever (for RRF-based fusion)
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangChainDocument
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)


def create_bm25_retriever(
    documents: List[LangChainDocument],
    k: int = 10,
    preprocess_func=None,
) -> BM25Retriever:
    """Create a LangChain BM25Retriever from documents.
    
    Args:
        documents: List of LangChain Document objects
        k: Number of documents to retrieve
        preprocess_func: Optional preprocessing function for text
        
    Returns:
        Configured BM25Retriever
    """
    kwargs = {"k": k}
    if preprocess_func:
        kwargs["preprocess_func"] = preprocess_func
    
    retriever = BM25Retriever.from_documents(documents, **kwargs)
    logger.info(f"Created BM25Retriever with {len(documents)} documents, k={k}")
    return retriever


def create_hybrid_retriever(
    dense_retriever,
    sparse_retriever: BM25Retriever,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> EnsembleRetriever:
    """Create a hybrid retriever using LangChain EnsembleRetriever with RRF.
    
    The EnsembleRetriever uses Reciprocal Rank Fusion (RRF) internally
    to combine results from multiple retrievers.
    
    Args:
        dense_retriever: Vector store retriever (e.g., Chroma.as_retriever())
        sparse_retriever: BM25Retriever instance
        dense_weight: Weight for dense retriever results (0-1)
        sparse_weight: Weight for sparse retriever results (0-1)
        
    Returns:
        Configured EnsembleRetriever with RRF fusion
    """
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight],
    )
    
    logger.info(
        f"Created EnsembleRetriever with weights: "
        f"dense={dense_weight}, sparse={sparse_weight}"
    )
    return ensemble_retriever


# ============================================================================
# Backward compatibility classes
# ============================================================================

class BM25Searcher:
    """Backward-compatible wrapper around LangChain BM25Retriever.
    
    Provides the same interface as the original custom BM25Searcher.
    """
    
    def __init__(self, k: int = 10):
        self.k = k
        self._retriever: Optional[BM25Retriever] = None
        self._documents: List[LangChainDocument] = []
    
    def index(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Index documents for BM25 search."""
        metadatas = metadatas or [{} for _ in texts]
        self._documents = [
            LangChainDocument(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self._retriever = create_bm25_retriever(self._documents, k=self.k)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search indexed documents."""
        if self._retriever is None:
            return []
        
        if top_k:
            self._retriever.k = top_k
        
        results = self._retriever.invoke(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 1.0 / (idx + 1),  # Approximate score from rank
            }
            for idx, doc in enumerate(results)
        ]


class HybridSearcher:
    """Backward-compatible wrapper around LangChain EnsembleRetriever.
    
    Provides the same interface as the original custom HybridSearcher.
    """
    
    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._ensemble: Optional[EnsembleRetriever] = None
    
    def setup(self, dense_retriever, sparse_retriever: BM25Retriever):
        """Set up the hybrid retriever."""
        self._ensemble = create_hybrid_retriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )
    
    def search(self, query: str) -> List[LangChainDocument]:
        """Perform hybrid search."""
        if self._ensemble is None:
            raise ValueError("HybridSearcher not set up. Call setup() first.")
        return self._ensemble.invoke(query)
