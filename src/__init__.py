"""Modular RAG - Production Ready RAG System using LangChain and LangGraph.

A modular, production-ready Retrieval-Augmented Generation system
built on LangChain and LangGraph with:
- Hybrid search (BM25Retriever + EnsembleRetriever with RRF)
- HyDE (HypotheticalDocumentEmbedder)
- RAPTOR (Recursive Abstractive Processing)
- Cross-encoder reranking (CrossEncoderReranker + ContextualCompressionRetriever)
- ColBERT reranking (custom late-interaction)
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
    get_llm,
    create_llm,
    LLMWrapper,
    get_embedding,
    create_embedding,
    EmbeddingWrapper,
    OllamaEmbedding,
)

from .components import (
    DocumentProcessor,
    ChunkingStrategy,
    RecursiveChunker,
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
    'get_llm',
    'create_llm',
    'LLMWrapper',
    'get_embedding',
    'create_embedding',
    'EmbeddingWrapper',
    'OllamaEmbedding',
    
    # Components
    'DocumentProcessor',
    'ChunkingStrategy',
    'RecursiveChunker',
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
