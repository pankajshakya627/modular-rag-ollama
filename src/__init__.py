"""Modular RAG - Production Ready RAG System.

A modular, production-ready Retrieval-Augmented Generation system
with support for HyDE, RAPTOR, hybrid search, and LangGraph orchestration.

Features:
- Modular architecture with reusable components
- Hybrid search (BM25 + Dense Vectors)
- HyDE (Hypothetical Document Embeddings)
- RAPTOR (Recursive Abstractive Processing)
- Multiple reranking methods (ColBERT, Cross-Encoder)
- LangGraph-based workflow orchestration
- FastAPI endpoints
- Comprehensive evaluation metrics

Usage:
    from modular_rag import ModularRAGWorkflow
    
    workflow = ModularRAGWorkflow()
    result = workflow.query("Your question here")
    print(result["answer"])
"""

__version__ = "1.0.0"
__author__ = "Modular RAG Team"

from .core import (
    get_config,
    Config,
    ConfigLoader,
    LLMWrapper,
    OllamaLLM,
    EmbeddingWrapper,
    OllamaEmbedding,
)

from .components import (
    DocumentProcessor,
    ChunkingStrategy,
    RecursiveChunker,
    SemanticChunker,
    VectorStoreManager,
    HybridSearcher,
    BM25Searcher,
    DenseRetriever,
    HyDERetriever,
    RAPTORRetriever,
    BaseReranker,
    ColBERTReranker,
    CrossEncoderReranker,
    AnswerGenerator,
    ResponseSynthesizer,
    RAGGraph,
    ModularRAGWorkflow,
    GraphState,
)

from .api import (
    app,
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse,
)

__all__ = [
    # Version
    '__version__',
    
    # Core
    'get_config',
    'Config',
    'ConfigLoader',
    'LLMWrapper',
    'OllamaLLM',
    'EmbeddingWrapper',
    'OllamaEmbedding',
    
    # Components
    'DocumentProcessor',
    'ChunkingStrategy',
    'RecursiveChunker',
    'SemanticChunker',
    'VectorStoreManager',
    'HybridSearcher',
    'BM25Searcher',
    'DenseRetriever',
    'HyDERetriever',
    'RAPTORRetriever',
    'BaseReranker',
    'ColBERTReranker',
    'CrossEncoderReranker',
    'AnswerGenerator',
    'ResponseSynthesizer',
    'RAGGraph',
    'ModularRAGWorkflow',
    'GraphState',
    
    # API
    'app',
    'QueryRequest',
    'QueryResponse',
    'IndexRequest',
    'IndexResponse',
    'HealthResponse',
]
