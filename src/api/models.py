"""API models for Modular RAG."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The user query to process")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous chat messages"
    )
    use_hyde: bool = Field(default=True, description="Use HyDE retrieval")
    use_raptor: bool = Field(default=True, description="Use RAPTOR retrieval")
    use_reranking: bool = Field(default=True, description="Use reranking")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    synthesis_strategy: str = Field(
        default="concatenate",
        description="Strategy for synthesizing responses"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used"
    )
    workflow_stage: str = Field(..., description="Current workflow stage")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    decomposed_queries: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-queries"
    )
    stepback_query: Optional[str] = Field(
        default=None,
        description="Step-back (broader) query"
    )
    processing_time: float = Field(..., description="Processing time in seconds")


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    file_path: Optional[str] = Field(
        default=None,
        description="Path to file to index"
    )
    directory_path: Optional[str] = Field(
        default=None,
        description="Path to directory to index"
    )
    text_content: Optional[str] = Field(
        default=None,
        description="Direct text content to index"
    )
    chunk_size: int = Field(default=1024, ge=100, le=10000)
    chunk_overlap: int = Field(default=128, ge=0, le=1000)
    chunking_strategy: str = Field(
        default="recursive",
        description="Strategy for chunking documents"
    )
    recursive: bool = Field(default=True, description="Recursive directory traversal")
    max_files: Optional[int] = Field(default=None, ge=1, description="Max files to index")


class IndexResponse(BaseModel):
    """Response model for indexing endpoint."""
    success: bool = Field(..., description="Indexing success status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    document_ids: List[str] = Field(
        default_factory=list,
        description="IDs of indexed documents"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


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
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    llm_available: bool = Field(..., description="LLM service availability")
    embedding_available: bool = Field(..., description="Embedding service availability")
    vector_store_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Vector store statistics"
    )


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    total_documents: int = Field(..., description="Total documents in store")
    total_chunks: int = Field(..., description="Total chunks in store")
    index_size_mb: float = Field(..., description="Index size in MB")


class DeleteRequest(BaseModel):
    """Request model for document deletion."""
    document_ids: List[str] = Field(..., description="IDs of documents to delete")


class DeleteResponse(BaseModel):
    """Response model for document deletion."""
    success: bool = Field(..., description="Deletion success status")
    documents_deleted: int = Field(..., description="Number of documents deleted")
    error: Optional[str] = Field(default=None, description="Error message if failed")
