"""Configuration loader for Modular RAG system."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = "Modular RAG"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama"
    model: str = "rnj-1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    context_length: int = 8192


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    provider: str = "ollama"
    model: str = "nomic-embed-text-v2-moe"
    base_url: str = "http://localhost:11434"
    dimensions: int = 768
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    provider: str = "chromadb"
    collection_name: str = "modular_rag_documents"
    persist_directory: str = "./data/vector_store"
    distance_metric: str = "cosine"


class BM25Config(BaseModel):
    """BM25 configuration."""
    enabled: bool = True
    k1: float = 1.2
    b: float = 0.75
    top_k: int = 20


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""
    enabled: bool = True
    fusion_method: str = "rrf"
    alpha: float = 0.5
    top_k: int = 50


class RerankingConfig(BaseModel):
    """Reranking configuration."""
    enabled: bool = True
    method: str = "colbert"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    colbert_max_tokens: int = 512
    top_k: int = 10
    batch_size: int = 32


class HyDEConfig(BaseModel):
    """HyDE configuration."""
    enabled: bool = True
    hypothesis_prompt: str = ""
    temperature: float = 0.3
    max_tokens: int = 512


class QueryDecompositionConfig(BaseModel):
    """Query decomposition configuration."""
    enabled: bool = True
    method: str = "rq-rag"
    max_sub_questions: int = 5
    decomposition_prompt: str = ""
    stepback_prompt: str = ""


class RAPTORConfig(BaseModel):
    """RAPTOR configuration."""
    enabled: bool = True
    clustering_method: str = "hierarchical"
    num_clusters: int = 5
    embedding_model: str = "nomic-embed-text-v2-moe"
    summarization_model: str = "rnj-1:8b"
    max_summary_length: int = 512
    threshold: float = 0.7


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: list = Field(default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""])
    length_function: str = "len"


class GranularityRetrievalConfig(BaseModel):
    """Granularity-aware retrieval configuration."""
    enabled: bool = True
    long_context_compression: bool = True
    compression_ratio: float = 0.5
    filter_irrelevant_spans: bool = True
    span_filter_threshold: float = 0.3


class LangGraphConfig(BaseModel):
    """LangGraph orchestration configuration."""
    enabled: bool = True
    checkpointing: bool = True
    human_in_loop: bool = False
    workflow_name: str = "modular_rag_workflow"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list = Field(default_factory=lambda: ["*"])
    docs_enabled: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    enabled: bool = True
    metrics: list = Field(default_factory=lambda: [
        "retrieval_precision", "retrieval_recall", "answer_relevancy", 
        "faithfulness", "context_precision"
    ])
    sample_size: int = 100


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    cache_type: str = "disk"
    max_size: int = 1000
    ttl: int = 3600


class ProductionConfig(BaseModel):
    """Production settings configuration."""
    environment: str = "development"
    monitoring: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30


class Config(BaseModel):
    """Main configuration class."""
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)
    query_decomposition: QueryDecompositionConfig = Field(default_factory=QueryDecompositionConfig)
    raptor: RAPTORConfig = Field(default_factory=RAPTORConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    granularity_retrieval: GranularityRetrievalConfig = Field(default_factory=GranularityRetrievalConfig)
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)


class ConfigLoader:
    """Configuration loader with environment variable support."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for config.yaml in standard locations
            possible_paths = [
                Path("config/config.yaml"),
                Path("config.yaml"),
                Path("../config/config.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._config = Config(**config_dict)
        else:
            # Use default configuration
            self._config = Config()
            # Set default prompts
            self._config.hyde.hypothesis_prompt = """Write a detailed hypothetical answer to the question. 
Include specific facts, examples, and reasoning that would appear in a relevant document.
Question: {query}
Hypothetical Answer:"""
            self._config.query_decomposition.decomposition_prompt = """Decompose the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.
Original Question: {query}
Sub-questions:"""
            self._config.query_decomposition.stepback_prompt = """Generate a broader, more general question that captures the context of the original query.
This helps retrieve relevant background information.
Original Question: {query}
Step-back Question:"""
    
    def get_config(self) -> Config:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file."""
        self._config = None
        self._load_config(config_path)


def get_config() -> Config:
    """Get the global configuration instance."""
    loader = ConfigLoader()
    return loader.get_config()


# Convenience function to get specific config sections
def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_config().llm


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration."""
    return get_config().embedding


def get_vector_store_config() -> VectorStoreConfig:
    """Get vector store configuration."""
    return get_config().vector_store


def get_hybrid_search_config() -> HybridSearchConfig:
    """Get hybrid search configuration."""
    return get_config().hybrid_search


def get_reranking_config() -> RerankingConfig:
    """Get reranking configuration."""
    return get_config().reranking


def get_hyde_config() -> HyDEConfig:
    """Get HyDE configuration."""
    return get_config().hyde


def get_query_decomposition_config() -> QueryDecompositionConfig:
    """Get query decomposition configuration."""
    return get_config().query_decomposition


def get_raptor_config() -> RAPTORConfig:
    """Get RAPTOR configuration."""
    return get_config().raptor
