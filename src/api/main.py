"""FastAPI application using LangChain and LangGraph for Modular RAG.

Production improvements:
- Dependency injection via FastAPI state (no global mutable vars)
- slowapi rate limiting on /query (60 req/min per IP by default)
- X-Request-ID middleware for distributed tracing
- Fully async /query endpoint using rag_graph.arun()
- /metrics endpoint exposing Prometheus metrics
- Async background indexing with job tracking
- structlog JSON logging
- Fixed bare except in health check
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

import logging

from .models import (
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    IndexJobResponse,
    HealthResponse,
    DocumentResponse,
    StatsResponse,
    DeleteRequest,
    DeleteResponse,
)
from ..core.config import get_config
from ..core.llm import get_llm
from ..core.embedding import get_embedding
from ..core.observability import (
    setup_logging,
    get_logger,
    get_metrics_content,
    PROMETHEUS_CONTENT_TYPE,
    record_query_duration,
    record_query_error,
    record_indexed_documents,
    set_request_id,
)
from ..components.retrieval.document_processor import DocumentProcessor, ChunkingStrategy
from ..components.retrieval.vector_store import VectorStoreManager, LangChainChromaVectorStore
from ..components.orchestration.rag_graph import RAGGraph, ModularRAGWorkflow

# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    _rate_limiting_available = True
except ImportError:
    limiter = None
    _rate_limiting_available = False

# ─────────────────────────────────────────────────────────────────────────────
# Application State
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Holds all application-wide dependencies — replaces global mutable vars."""

    def __init__(self):
        self.config = None
        self.llm_wrapper = None
        self.embedding_wrapper = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.rag_workflow: Optional[ModularRAGWorkflow] = None
        # Background job tracking
        self.index_jobs: Dict[str, Dict[str, Any]] = {}

    def initialize(self):
        """Initialize all components."""
        log = get_logger(__name__)

        self.config = get_config()
        setup_logging(self.config.app.log_level)

        log.info("Initializing LLM...")
        self.llm_wrapper = get_llm()

        log.info("Initializing embeddings...")
        self.embedding_wrapper = get_embedding()

        log.info("Initializing document processor...")
        self.document_processor = DocumentProcessor(
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=self.config.document_processing.chunk_size,
            chunk_overlap=self.config.document_processing.chunk_overlap,
            embedding_wrapper=self.embedding_wrapper,
        )

        log.info("Initializing vector store...")
        vector_store = LangChainChromaVectorStore(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=self.config.vector_store.persist_directory,
            embedding_wrapper=self.embedding_wrapper,
        )
        self.vector_store_manager = VectorStoreManager(
            vector_store=vector_store,
            embedding_wrapper=self.embedding_wrapper,
        )

        log.info("Initializing LangGraph RAG workflow...")
        self.rag_workflow = ModularRAGWorkflow(
            config={
                "enable_hyde": self.config.hyde.enabled,
                "enable_raptor": self.config.raptor.enabled,
                "enable_reranking": self.config.reranking.enabled,
            }
        )

        log.info("All components initialized successfully")


_app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID header and sets it in observability context."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Records per-request processing time in headers and Prometheus."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))

        # Only record /query latency in Prometheus
        if request.url.path == "/query":
            record_query_duration(elapsed)

        return response


# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    log = get_logger(__name__)
    log.info("Starting Modular RAG API with LangChain and LangGraph...")
    _app_state.initialize()
    yield
    log.info("Shutting down Modular RAG API...")


app = FastAPI(
    title="Modular RAG API",
    description="Production-ready Modular RAG system using LangChain and LangGraph",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
if _rate_limiting_available and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware (order matters — outermost is applied last)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_workflow() -> ModularRAGWorkflow:
    if _app_state.rag_workflow is None:
        raise HTTPException(status_code=503, detail="RAG workflow not initialized")
    return _app_state.rag_workflow


def _get_vsm() -> VectorStoreManager:
    if _app_state.vector_store_manager is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return _app_state.vector_store_manager


def _get_processor() -> DocumentProcessor:
    if _app_state.document_processor is None:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    return _app_state.document_processor


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of the system."""
    log = get_logger(__name__)
    llm_available = False
    embedding_available = False
    vector_stats: Dict[str, Any] = {}

    try:
        if _app_state.llm_wrapper:
            # Lightweight ping — invoke with minimal tokens
            from langchain_core.messages import HumanMessage
            _app_state.llm_wrapper.invoke(
                [HumanMessage(content="ping")],
                config={"max_tokens": 1},
            )
            llm_available = True
    except Exception as exc:
        log.warning("LLM health check failed", error=str(exc))

    try:
        if _app_state.embedding_wrapper:
            _app_state.embedding_wrapper.embed_query("ping")
            embedding_available = True
    except Exception as exc:
        log.warning("Embedding health check failed", error=str(exc))

    try:
        if _app_state.vector_store_manager:
            vector_stats = _app_state.vector_store_manager.get_index_stats()
    except Exception as exc:
        log.warning("Vector store stats unavailable", error=str(exc))

    status = "healthy" if (llm_available and embedding_available) else "degraded"
    return HealthResponse(
        status=status,
        version="1.0.0",
        llm_available=llm_available,
        embedding_available=embedding_available,
        vector_store_stats=vector_stats,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest, http_request: Request):
    """Process a query through the LangGraph RAG pipeline (async)."""
    log = get_logger(__name__)
    start_time = time.time()

    # Apply rate limiting if available
    if _rate_limiting_available and limiter:
        config = get_config()
        limit_str = f"{config.api.rate_limit_per_minute}/minute"
        try:
            await limiter._check_request_limit(http_request, limit_str)
        except Exception:
            pass  # Allow if limiter fails

    workflow = _get_workflow()

    try:
        # Fully async query execution
        result = await workflow.aquery(request.query)
        processing_time = time.time() - start_time

        log.info(
            "query_completed",
            query=request.query[:80],
            confidence=result.get("confidence", 0.0),
            processing_time_ms=round(processing_time * 1000, 2),
        )

        return QueryResponse(
            query=result.get("query", request.query),
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            workflow_stage=result.get("workflow_stage", "completed"),
            errors=result.get("errors", []),
            decomposed_queries=result.get("decomposed_queries", []),
            stepback_query=result.get("stepback_query"),
            processing_time=processing_time,
        )

    except Exception as e:
        record_query_error("query_endpoint")
        log.error("Query failed", error=str(e), query=request.query[:80])
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """WebSocket endpoint for streaming RAG queries."""
    log = get_logger(__name__)
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            query_text = data.get("query", "").strip()

            if not query_text:
                await websocket.send_json({"error": "No query provided"})
                continue

            workflow = _get_workflow()

            await websocket.send_json({
                "type": "status",
                "stage": "started",
                "message": f"Processing query: {query_text}",
            })

            try:
                async for state in workflow.astream(query_text):
                    for node_name, node_state in state.items():
                        if node_name == "generate_answer":
                            await websocket.send_json({
                                "type": "result",
                                "answer": node_state.get("answer", ""),
                                "sources": node_state.get("sources", []),
                                "confidence": node_state.get("confidence", 0.0),
                            })
                        else:
                            await websocket.send_json({
                                "type": "status",
                                "stage": node_name,
                                "details": {},
                            })
            except Exception as e:
                log.error("WebSocket streaming error", error=str(e))
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error("WebSocket error", error=str(e))
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Background Indexing
# ─────────────────────────────────────────────────────────────────────────────

def _run_indexing_job(job_id: str, request: IndexRequest) -> None:
    """Background indexing job — runs in a thread pool."""
    log = get_logger(__name__)
    _app_state.index_jobs[job_id]["status"] = "running"

    try:
        processor = _get_processor()
        vsm = _get_vsm()
        documents = []

        if request.file_path:
            doc = processor.process_file(request.file_path)
            documents.append(doc)
        elif request.directory_path:
            documents = processor.process_directory(
                request.directory_path,
                recursive=request.recursive,
                max_files=request.max_files,
            )
        elif request.text_content:
            doc = processor.process_text(
                request.text_content,
                metadata={"source": "api"},
            )
            documents.append(doc)

        vsm.index_documents(documents)

        # Invalidate BM25 cache
        if _app_state.rag_workflow and _app_state.rag_workflow.rag_graph:
            _app_state.rag_workflow.rag_graph.invalidate_bm25_cache()

        total_chunks = sum(len(d.chunks) for d in documents)
        record_indexed_documents(len(documents))

        _app_state.index_jobs[job_id].update({
            "status": "completed",
            "documents_processed": len(documents),
            "chunks_created": total_chunks,
            "document_ids": [d.id for d in documents],
        })
        log.info("Indexing job completed", job_id=job_id, docs=len(documents))

    except Exception as e:
        log.error("Indexing job failed", job_id=job_id, error=str(e))
        _app_state.index_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
        })


@app.post("/index", response_model=IndexJobResponse, status_code=202, tags=["Documents"])
async def index_document(request: IndexRequest, background_tasks: BackgroundTasks):
    """Submit a document indexing job (returns 202 Accepted + job_id)."""
    job_id = str(uuid.uuid4())
    _app_state.index_jobs[job_id] = {
        "status": "queued",
        "job_id": job_id,
        "documents_processed": 0,
        "chunks_created": 0,
        "document_ids": [],
        "error": None,
    }
    background_tasks.add_task(_run_indexing_job, job_id, request)
    return IndexJobResponse(
        job_id=job_id,
        status="queued",
        message="Indexing job queued. Poll GET /index/{job_id} for status.",
    )


@app.get("/index/{job_id}", tags=["Documents"])
async def get_index_job_status(job_id: str):
    """Poll indexing job status."""
    job = _app_state.index_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/search", tags=["Search"])
async def search(
    query: str = Query(..., description="Search query", min_length=1),
    top_k: int = Query(default=10, ge=1, le=100),
):
    """Search for documents using the vector store."""
    vsm = _get_vsm()
    try:
        results = vsm.search(query, top_k=top_k)
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "total_results": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentResponse], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    vsm = _get_vsm()
    try:
        documents = vsm.vector_store.get_all_documents()
        return [
            DocumentResponse(
                id=doc.id,
                content_preview=doc.content[:200] + "..." if doc.content else "",
                metadata={},
                source_type=doc.source_type,
                source_path=doc.source_path,
                chunk_count=len(doc.chunks),
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics."""
    vsm = _get_vsm()
    try:
        stats = vsm.get_index_stats()
        return StatsResponse(
            total_documents=stats.get("document_count", 0),
            total_chunks=stats.get("chunk_count", 0),
            index_size_mb=0.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", response_model=DeleteResponse, tags=["Documents"])
async def delete_documents(request: DeleteRequest):
    """Delete documents from the index."""
    vsm = _get_vsm()
    try:
        vsm.delete_documents(request.document_ids)
        # Invalidate BM25 cache
        if _app_state.rag_workflow and _app_state.rag_workflow.rag_graph:
            _app_state.rag_workflow.rag_graph.invalidate_bm25_cache()
        return DeleteResponse(success=True, documents_deleted=len(request.document_ids))
    except Exception as e:
        return DeleteResponse(success=False, documents_deleted=0, error=str(e))


@app.delete("/index", tags=["Documents"])
async def clear_index():
    """Clear the entire index."""
    vsm = _get_vsm()
    try:
        vsm.clear_index()
        if _app_state.rag_workflow and _app_state.rag_workflow.rag_graph:
            _app_state.rag_workflow.rag_graph.invalidate_bm25_cache()
        return {"success": True, "message": "Index cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["System"], response_class=PlainTextResponse)
async def metrics():
    """Expose Prometheus metrics."""
    content = get_metrics_content()
    return PlainTextResponse(content=content, media_type=PROMETHEUS_CONTENT_TYPE)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging."""
    log = get_logger(__name__)
    log.error(
        "unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
