"""FastAPI application using LangChain and LangGraph for Modular RAG."""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
import logging

from .models import (
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse,
    DocumentResponse,
    StatsResponse,
    DeleteRequest,
    DeleteResponse,
)
from ..core.config import get_config
from ..core.llm import LLMWrapper, get_llm
from ..core.embedding import EmbeddingWrapper, get_embedding
from ..components.retrieval.document_processor import DocumentProcessor, ChunkingStrategy
from ..components.retrieval.vector_store import VectorStoreManager, LangChainChromaVectorStore
from ..components.orchestration.rag_graph import RAGGraph, ModularRAGWorkflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global instances
_config = None
_llm_wrapper: Optional[LLMWrapper] = None
_embedding_wrapper: Optional[EmbeddingWrapper] = None
_document_processor: Optional[DocumentProcessor] = None
_vector_store_manager: Optional[VectorStoreManager] = None
_rag_workflow: Optional[ModularRAGWorkflow] = None


class APICallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for API logging."""
    
    def __init__(self):
        self.logs = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logs.append(f"LLM started with prompts: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        self.logs.append(f"LLM completed with response")
    
    def on_retriever_start(self, query, **kwargs):
        self.logs.append(f"Retriever started with query: {query}")
    
    def on_retriever_end(self, documents, **kwargs):
        self.logs.append(f"Retriever completed with {len(documents)} documents")


def init_components():
    """Initialize global components with LangChain."""
    global _config, _llm_wrapper, _embedding_wrapper
    global _document_processor, _vector_store_manager, _rag_workflow
    
    try:
        # Load config
        _config = get_config()
        
        # Initialize LLM with LangChain
        _llm_wrapper = get_llm()
        
        # Initialize embedding with LangChain
        _embedding_wrapper = get_embedding()
        
        # Initialize document processor
        _document_processor = DocumentProcessor(
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=_config.document_processing.chunk_size,
            chunk_overlap=_config.document_processing.chunk_overlap,
            embedding_wrapper=_embedding_wrapper,
        )
        
        # Initialize LangChain ChromaDB vector store
        vector_store = LangChainChromaVectorStore(
            collection_name=_config.vector_store.collection_name,
            persist_directory=_config.vector_store.persist_directory,
            embedding_wrapper=_embedding_wrapper,
        )
        _vector_store_manager = VectorStoreManager(
            vector_store=vector_store,
            embedding_wrapper=_embedding_wrapper,
        )
        
        # Initialize LangGraph RAG workflow
        _rag_workflow = ModularRAGWorkflow(
            config={
                "enable_hyde": _config.hyde.enabled,
                "enable_raptor": _config.raptor.enabled,
                "enable_reranking": _config.reranking.enabled,
            }
        )
        
        logger.info("All LangChain and LangGraph components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for FastAPI."""
    # Startup
    logger.info("Starting Modular RAG API with LangChain and LangGraph...")
    init_components()
    yield
    # Shutdown
    logger.info("Shutting down Modular RAG API...")


# Create FastAPI app with LangChain integration
app = FastAPI(
    title="Modular RAG API",
    description="Production-ready Modular RAG system using LangChain and LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check the health of the LangChain/LangGraph system."""
    llm_available = False
    embedding_available = False
    
    # Check LLM availability with LangChain
    try:
        if _llm_wrapper:
            response = _llm_wrapper.llm.invoke("test")
            llm_available = True
    except Exception:
        pass
    
    # Check embedding availability with LangChain
    try:
        if _embedding_wrapper:
            _embedding_wrapper.embed_query("test")
            embedding_available = True
    except Exception:
        pass
    
    # Get vector store stats
    vector_stats = {}
    try:
        if _vector_store_manager:
            vector_stats = _vector_store_manager.get_index_stats()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if (llm_available and embedding_available) else "degraded",
        version="1.0.0",
        llm_available=llm_available,
        embedding_available=embedding_available,
        vector_store_stats=vector_stats,
    )


# Query endpoint using LangGraph workflow
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """Process a query through the LangGraph RAG pipeline."""
    start_time = time.time()
    
    try:
        if _rag_workflow is None:
            raise HTTPException(status_code=503, detail="LangGraph workflow not initialized")
        
        # Run the LangGraph workflow
        result = _rag_workflow.query(request.query)
        
        processing_time = time.time() - start_time
        
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
        logger.error(f"Error processing query with LangGraph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Index endpoint
@app.post("/index", response_model=IndexResponse, tags=["Documents"])
async def index_document(request: IndexRequest):
    """Index a document or directory using LangChain."""
    try:
        if _document_processor is None or _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Indexing components not initialized")
        
        documents = []
        
        if request.file_path:
            # Index single file
            doc = _document_processor.process_file(request.file_path)
            documents.append(doc)
            
        elif request.directory_path:
            # Index directory
            documents = _document_processor.process_directory(
                request.directory_path,
                recursive=request.recursive,
                max_files=request.max_files,
            )
            
        elif request.text_content:
            # Index direct text content
            doc = _document_processor.process_text(
                request.text_content,
                metadata={"source": "api"},
            )
            documents.append(doc)
            
        else:
            raise HTTPException(status_code=400, detail="No content provided to index")
        
        # Update chunking strategy if specified
        if request.chunking_strategy != "recursive":
            strategy_map = {
                "semantic": ChunkingStrategy.SEMANTIC,
                "fixed_size": ChunkingStrategy.FIXED_SIZE,
                "sentence": ChunkingStrategy.SENTENCE,
            }
            strategy = strategy_map.get(request.chunking_strategy, ChunkingStrategy.RECURSIVE)
            _document_processor.set_chunking_strategy(strategy)
        
        # Index documents using LangChain vector store
        _vector_store_manager.index_documents(documents)
        
        total_chunks = sum(len(doc.chunks) for doc in documents)
        
        return IndexResponse(
            success=True,
            documents_processed=len(documents),
            chunks_created=total_chunks,
            document_ids=[doc.id for doc in documents],
        )
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return IndexResponse(
            success=False,
            documents_processed=0,
            chunks_created=0,
            error=str(e),
        )


# Search endpoint using LangChain vector store
@app.get("/search", tags=["Search"])
async def search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
):
    """Search for documents using LangChain vector store."""
    try:
        if _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        results = _vector_store_manager.search(query, top_k=top_k)
        
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "total_results": len(results),
        }
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Documents list endpoint
@app.get("/documents", response_model=list[DocumentResponse], tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    try:
        if _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        documents = _vector_store_manager.vector_store.get_all_documents()
        
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
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Stats endpoint
@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics."""
    try:
        if _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        stats = _vector_store_manager.get_index_stats()
        
        return StatsResponse(
            total_documents=stats.get("document_count", 0),
            total_chunks=stats.get("chunk_count", 0),
            index_size_mb=0.0,  # Would need to calculate actual size
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete endpoint
@app.delete("/documents", response_model=DeleteResponse, tags=["Documents"])
async def delete_documents(request: DeleteRequest):
    """Delete documents from the index."""
    try:
        if _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        _vector_store_manager.delete_documents(request.document_ids)
        
        return DeleteResponse(
            success=True,
            documents_deleted=len(request.document_ids),
        )
        
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        return DeleteResponse(
            success=False,
            documents_deleted=0,
            error=str(e),
        )


# Clear index endpoint
@app.delete("/index", tags=["Documents"])
async def clear_index():
    """Clear the entire index."""
    try:
        if _vector_store_manager is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        _vector_store_manager.clear_index()
        
        return {"success": True, "message": "Index cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
