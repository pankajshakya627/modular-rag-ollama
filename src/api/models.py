"""API models for Modular RAG — Pydantic v2 with full validation."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What were the Q3 2024 revenue results?",
                "use_hyde": True,
                "use_raptor": True,
                "use_reranking": True,
                "top_k": 10,
            }
        }
    )

    query: str = Field(..., min_length=1, description="The user query to process")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous chat messages as list of {role, content} dicts",
    )
    use_hyde: bool = Field(default=True, description="Use HyDE retrieval")
    use_raptor: bool = Field(default=True, description="Use RAPTOR retrieval")
    use_reranking: bool = Field(default=True, description="Use reranking")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    synthesis_strategy: Literal["concatenate", "merge", "best"] = Field(
        default="merge",
        description="Strategy for synthesizing multiple sub-query responses",
    )

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace-only")
        return v


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence score")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list, description="Source documents used"
    )
    workflow_stage: str = Field(..., description="Final workflow stage reached")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    decomposed_queries: List[str] = Field(
        default_factory=list, description="Decomposed sub-queries"
    )
    stepback_query: Optional[str] = Field(
        default=None, description="Step-back (broader) query"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "/data/reports/Q3_2024.pdf",
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "chunking_strategy": "recursive",
            }
        }
    )

    file_path: Optional[str] = Field(default=None, description="Path to file to index")
    directory_path: Optional[str] = Field(
        default=None, description="Path to directory to index"
    )
    text_content: Optional[str] = Field(
        default=None, description="Direct text content to index"
    )
    chunk_size: int = Field(default=1024, ge=100, le=10000)
    chunk_overlap: int = Field(default=128, ge=0, le=1000)
    chunking_strategy: Literal["recursive", "semantic", "fixed_size", "sentence"] = Field(
        default="recursive",
        description="Strategy for chunking documents",
    )
    recursive: bool = Field(default=True, description="Recursive directory traversal")
    max_files: Optional[int] = Field(default=None, ge=1, description="Max files to index")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1024)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class IndexResponse(BaseModel):
    """Response model for synchronous indexing (deprecated — use IndexJobResponse)."""
    success: bool = Field(..., description="Indexing success status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    document_ids: List[str] = Field(
        default_factory=list, description="IDs of indexed documents"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class IndexJobResponse(BaseModel):
    """Response model for async indexing — returned immediately with job_id."""
    job_id: str = Field(..., description="Unique job identifier for polling")
    status: Literal["queued", "running", "completed", "failed"] = Field(
        ..., description="Current job status"
    )
    message: str = Field(..., description="Human-readable status message")


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str = Field(..., description="Document ID")
    content_preview: str = Field(..., description="Preview of document content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = Field(..., description="Source file type")
    source_path: Optional[str] = Field(default=None, description="Source file path")
    chunk_count: int = Field(..., description="Number of chunks")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Service status"
    )
    version: str = Field(..., description="Application version")
    llm_available: bool = Field(..., description="LLM service availability")
    embedding_available: bool = Field(..., description="Embedding service availability")
    vector_store_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Vector store statistics"
    )


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    total_documents: int = Field(..., description="Total documents in store")
    total_chunks: int = Field(..., description="Total chunks in store")
    index_size_mb: float = Field(..., description="Index size in MB")


class DeleteRequest(BaseModel):
    """Request model for document deletion."""
    document_ids: List[str] = Field(
        ..., min_length=1, description="IDs of documents to delete"
    )


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    success: bool = Field(..., description="Deletion success status")
    documents_deleted: int = Field(..., description="Number of documents deleted")
    error: Optional[str] = Field(default=None, description="Error message if failed")
