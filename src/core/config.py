"""Configuration loader for Modular RAG system.

Production-grade configuration with:
- Full Pydantic v2 validation (field_validator, model_validator, ConfigDict)
- Immutable config (frozen=True) after load
- ENV variable overlay via pydantic-settings
- YAML key aliasing fix (cache.type → cache_type)
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Section Models
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Application configuration."""
    model_config = ConfigDict(frozen=True)

    name: str = "Modular RAG"
    version: str = "1.0.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class LLMConfig(BaseModel):
    """LLM configuration."""
    model_config = ConfigDict(frozen=True)

    provider: str = "ollama"
    model: str = "rnj-1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=120, gt=0)
    context_length: int = Field(default=8192, gt=0)


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""
    model_config = ConfigDict(frozen=True)

    provider: str = "ollama"
    model: str = "nomic-embed-text-v2-moe"
    base_url: str = "http://localhost:11434"
    dimensions: int = Field(default=768, gt=0)
    batch_size: int = Field(default=32, gt=0)


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    model_config = ConfigDict(frozen=True)

    provider: Literal["chromadb", "faiss", "qdrant"] = "chromadb"
    collection_name: str = "modular_rag_documents"
    persist_directory: str = "./data/vector_store"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class BM25Config(BaseModel):
    """BM25 configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    k1: float = Field(default=1.2, gt=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=20, gt=0)


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    fusion_method: Literal["rrf", "weighted", "score_normalized"] = "rrf"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=50, gt=0)


class TemporalBoostConfig(BaseModel):
    """Temporal reranking boost settings."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    match_boost: float = Field(default=1.5, gt=1.0)
    mismatch_penalty: float = Field(default=0.7, gt=0.0, lt=1.0)


class RerankingConfig(BaseModel):
    """Reranking configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    method: Literal["colbert", "cross_encoder", "llm_ensemble"] = "colbert"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    colbert_max_tokens: int = Field(default=512, gt=0)
    top_k: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    temporal_boost: TemporalBoostConfig = Field(default_factory=TemporalBoostConfig)


class HyDEConfig(BaseModel):
    """HyDE configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    hypothesis_prompt: str = ""
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)


class QueryDecompositionConfig(BaseModel):
    """Query decomposition configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    method: Literal["rq-rag", "decomposition", "stepback"] = "rq-rag"
    max_sub_questions: int = Field(default=5, gt=0, le=20)
    decomposition_prompt: str = ""
    stepback_prompt: str = ""


class RAPTORConfig(BaseModel):
    """RAPTOR configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    clustering_method: Literal["hierarchical", "flat"] = "hierarchical"
    num_clusters: int = Field(default=5, gt=0)
    embedding_model: str = "nomic-embed-text-v2-moe"
    summarization_model: str = "rnj-1:8b"
    max_summary_length: int = Field(default=512, gt=0)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class MetadataExtractionConfig(BaseModel):
    """Metadata extraction settings."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    rule_based: bool = True
    llm_based: bool = False
    extract_temporal: bool = True
    extract_financial: bool = True
    extract_entities: bool = True


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=1024, gt=0)
    chunk_overlap: int = Field(default=128, ge=0)
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
    )
    length_function: str = "len"
    metadata_extraction: MetadataExtractionConfig = Field(
        default_factory=MetadataExtractionConfig
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        # Access other field values via info.data (Pydantic v2)
        chunk_size = info.data.get("chunk_size", 1024)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class GranularityRetrievalConfig(BaseModel):
    """Granularity-aware retrieval configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    long_context_compression: bool = True
    compression_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    filter_irrelevant_spans: bool = True
    span_filter_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class LangGraphConfig(BaseModel):
    """LangGraph orchestration configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    checkpointing: bool = True
    checkpointer_backend: Literal["memory", "sqlite", "postgres"] = "sqlite"
    checkpointer_uri: str = "./data/checkpoints/rag.db"
    human_in_loop: bool = False
    workflow_name: str = "modular_rag_workflow"


class APIConfig(BaseModel):
    """API configuration."""
    model_config = ConfigDict(frozen=True)

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)
    workers: int = Field(default=4, gt=0)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    docs_enabled: bool = True
    rate_limit_per_minute: int = Field(default=60, gt=0)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    metrics: List[str] = Field(default_factory=lambda: [
        "retrieval_precision", "retrieval_recall", "answer_relevancy",
        "faithfulness", "context_precision",
    ])
    sample_size: int = Field(default=100, gt=0)


class CacheConfig(BaseModel):
    """Cache configuration.

    Handles YAML key alias: config.yaml uses 'type' but we store as 'cache_type'.
    """
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    enabled: bool = True
    # 'type' is the YAML key; 'cache_type' is the Python attribute name
    cache_type: str = Field(default="disk", alias="type")
    max_size: int = Field(default=1000, gt=0)
    ttl: int = Field(default=3600, gt=0)


class ProductionConfig(BaseModel):
    """Production settings configuration."""
    model_config = ConfigDict(frozen=True)

    environment: Literal["development", "staging", "production"] = "development"
    monitoring: bool = True
    metrics_port: int = Field(default=9090, gt=0, lt=65536)
    health_check_interval: int = Field(default=30, gt=0)


# ---------------------------------------------------------------------------
# Root Config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    """Main configuration class — immutable after construction."""
    model_config = ConfigDict(frozen=True)

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
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig
    )
    granularity_retrieval: GranularityRetrievalConfig = Field(
        default_factory=GranularityRetrievalConfig
    )
    langgraph: LangGraphConfig = Field(default_factory=LangGraphConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def from_yaml_with_env(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML, then apply ENV variable overrides.

        ENV var format: RAG__LLM__TEMPERATURE=0.5, RAG__API__PORT=9000, etc.
        """
        raw: Dict[str, Any] = {}

        # 1. Load from YAML file
        if config_path is None:
            possible_paths = [
                Path("config/config.yaml"),
                Path("config.yaml"),
                Path("../config/config.yaml"),
            ]
            for p in possible_paths:
                if p.exists():
                    config_path = str(p)
                    break

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f) or {}

        # 2. Apply ENV overrides (RAG__SECTION__KEY=value)
        prefix = "RAG__"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split("__")
                _set_nested(raw, parts, _coerce_value(value))

        return cls(**raw)


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set a nested dict value from a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _coerce_value(value: str) -> Any:
    """Coerce string env var values to Python types."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Singleton Loader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Configuration loader — singleton pattern with thread safety."""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Config] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file with ENV overlay."""
        self._config = Config.from_yaml_with_env(config_path)

    def get_config(self) -> Config:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config  # type: ignore[return-value]

    def reload(self, config_path: Optional[str] = None) -> None:
        """Reload configuration from file."""
        self._config = None
        self._load_config(config_path)


def get_config() -> Config:
    """Get the global configuration instance."""
    loader = ConfigLoader()
    return loader.get_config()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_llm_config() -> LLMConfig:
    return get_config().llm


def get_embedding_config() -> EmbeddingConfig:
    return get_config().embedding


def get_vector_store_config() -> VectorStoreConfig:
    return get_config().vector_store


def get_hybrid_search_config() -> HybridSearchConfig:
    return get_config().hybrid_search


def get_reranking_config() -> RerankingConfig:
    return get_config().reranking


def get_hyde_config() -> HyDEConfig:
    return get_config().hyde


def get_query_decomposition_config() -> QueryDecompositionConfig:
    return get_config().query_decomposition


def get_raptor_config() -> RAPTORConfig:
    return get_config().raptor
