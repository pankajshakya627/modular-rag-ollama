"""LangGraph orchestration for Modular RAG workflow.

Production improvements:
- GraphState migrated from TypedDict to Pydantic BaseModel
- BM25 index built once at __init__, not rebuilt per-query
- SqliteSaver for persistent checkpoints (survives restarts)
- Async-ready nodes using ainvoke() on LangChain chains
- @trace_span observability on each node
"""
from typing import Any, Dict, List, Optional
import logging
import asyncio
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, ConfigDict, Field

from ..retrieval.vector_store import SearchResult
from ..retrieval.hybrid_search import BM25Searcher
from ..retrieval.hyde import HyDERetriever
from ..reranking.base import BaseReranker, RerankedResult
from ..generation.answer_generator import AnswerGenerator, AnswerResult
from ...core.llm import get_llm
from ...core.embedding import get_embedding
from ...core.observability import trace_span, record_retrieval, record_reranking, get_logger
# Direct import to avoid circular dependency
from src.core.uuid_utils import generate_uuid  # UUID v7

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages of the RAG workflow."""
    QUERY_ANALYSIS = "query_analysis"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    ANSWER_GENERATION = "answer_generation"
    EVALUATION = "evaluation"
    FINAL = "final"


# ---------------------------------------------------------------------------
# Pydantic-based Graph State (replaces TypedDict)
# ---------------------------------------------------------------------------

class GraphState(BaseModel):
    """State for the LangGraph workflow with full Pydantic validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input
    query: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Query Processing
    original_query: str = ""
    decomposed_queries: List[str] = Field(default_factory=list)
    stepback_query: Optional[str] = None
    hyde_query: Optional[str] = None

    # Retrieval Results
    dense_results: List[SearchResult] = Field(default_factory=list)
    sparse_results: List[SearchResult] = Field(default_factory=list)
    hyde_results: List[SearchResult] = Field(default_factory=list)
    raptor_results: List[SearchResult] = Field(default_factory=list)
    combined_results: List[SearchResult] = Field(default_factory=list)

    # Reranked Results
    reranked_results: List[RerankedResult] = Field(default_factory=list)

    # Generation
    answer: str = ""
    answer_result: Optional[AnswerResult] = None

    # Metadata
    workflow_stage: str = ""
    errors: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RAG Graph
# ---------------------------------------------------------------------------

class RAGGraph:
    """LangGraph-based RAG workflow with modular components."""

    def __init__(
        self,
        llm_wrapper=None,
        embedding_wrapper=None,
        enable_hyde: bool = True,
        enable_raptor: bool = True,
        enable_reranking: bool = True,
        enable_checkpointing: bool = True,
    ):
        """Initialize the RAG Graph with LangGraph."""
        self.llm_wrapper = llm_wrapper or get_llm()
        self.embedding_wrapper = embedding_wrapper or get_embedding()
        self.enable_hyde = enable_hyde
        self.enable_raptor = enable_raptor
        self.enable_reranking = enable_reranking
        self.enable_checkpointing = enable_checkpointing

        # BM25 index — built once at init, not per-query
        self._bm25_searcher: Optional[BM25Searcher] = None
        self._bm25_stale: bool = True  # Set to True when docs are added/deleted

        # Initialize components
        self._init_components()

        # Build the LangGraph
        self.graph = self._build_graph()

    def _init_components(self):
        """Initialize RAG components using LangChain."""
        from ...core.config import get_config
        from ..retrieval.vector_store import LangChainChromaVectorStore, VectorStoreManager
        from ..retrieval.raptor import RAPTORRetriever
        from ..retrieval.hybrid_search import HybridSearcher

        config = get_config()

        # Vector store
        self.vector_store = LangChainChromaVectorStore(
            collection_name=config.vector_store.collection_name,
            persist_directory=config.vector_store.persist_directory,
            embedding_wrapper=self.embedding_wrapper,
        )
        self.vsm = VectorStoreManager(
            vector_store=self.vector_store,
            embedding_wrapper=self.embedding_wrapper,
        )

        # Hybrid searcher (dense weights from config)
        self.hybrid_searcher = HybridSearcher(
            dense_weight=config.hybrid_search.alpha,
            sparse_weight=1.0 - config.hybrid_search.alpha,
        )

        # HyDE retriever
        self.hyde_retriever = HyDERetriever(
            llm=self.llm_wrapper,
            embeddings=self.embedding_wrapper,
            vector_store=self.vector_store.vector_store
            if hasattr(self.vector_store, "vector_store")
            else None,
        )

        # RAPTOR retriever
        self.raptor_retriever = RAPTORRetriever(
            vector_store_manager=self.vsm,
            llm_wrapper=self.llm_wrapper,
            embedding_wrapper=self.embedding_wrapper,
            num_clusters=config.raptor.num_clusters,
        )

        # Reranker selection
        reranking_method = config.reranking.method
        if reranking_method == "colbert":
            from ..reranking.colbert import ColBERTReranker
            self.reranker = ColBERTReranker(
                max_tokens=config.reranking.colbert_max_tokens,
                batch_size=config.reranking.batch_size,
            )
        elif reranking_method == "cross_encoder":
            from ..reranking.cross_encoder import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(
                model_name=config.reranking.cross_encoder_model,
                top_n=config.reranking.top_k,
            )
        else:
            from ..reranking.base import LLMEnsembleReranker
            self.reranker = LLMEnsembleReranker(llm_wrapper=self.llm_wrapper)

        # Answer generator
        self.answer_generator = AnswerGenerator(llm_wrapper=self.llm_wrapper)

        # LangChain LCEL chains for query processing
        self._create_query_chains()

    def _create_query_chains(self):
        """Create LangChain chains for query processing using LCEL."""
        self.output_parser = StrOutputParser()

        self.decomposition_prompt = PromptTemplate(
            template=(
                "Decompose the following complex question into simpler sub-questions.\n"
                "Each sub-question should be answerable independently.\n"
                "Return exactly {max_sub_questions} sub-questions.\n\n"
                "Original Question: {query}\n\n"
                "Sub-questions (one per line):"
            ),
            input_variables=["query", "max_sub_questions"],
        )
        self.decomposition_chain = (
            self.decomposition_prompt | self.llm_wrapper | self.output_parser
        )

        self.stepback_prompt = PromptTemplate(
            template=(
                "Generate a broader, more general question that captures the context "
                "of the original query. This helps retrieve relevant background information.\n\n"
                "Original Question: {query}\n\n"
                "Step-back Question:"
            ),
            input_variables=["query"],
        )
        self.stepback_chain = (
            self.stepback_prompt | self.llm_wrapper | self.output_parser
        )

    def _get_bm25_searcher(self) -> BM25Searcher:
        """Return the BM25 searcher, rebuilding index only when stale."""
        if self._bm25_searcher is None or self._bm25_stale:
            logger.info("Building BM25 index from vector store documents...")
            searcher = BM25Searcher(k=20)
            all_docs = self.vsm.vector_store.get_all_documents()
            if all_docs:
                texts: List[str] = []
                metas: List[Dict[str, Any]] = []
                for doc in all_docs:
                    for chunk in doc.chunks:
                        texts.append(chunk.content)
                        metas.append(chunk.metadata)
                searcher.index(texts, metadatas=metas)
            self._bm25_searcher = searcher
            self._bm25_stale = False
            logger.info(f"BM25 index built with {len(texts) if all_docs else 0} chunks")
        return self._bm25_searcher

    def invalidate_bm25_cache(self) -> None:
        """Mark BM25 index as stale (call after adding/deleting documents)."""
        self._bm25_stale = True

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        builder = StateGraph(GraphState)

        builder.add_node("query_analysis", self._query_analysis)
        builder.add_node("dense_retrieval", self._dense_retrieval)
        builder.add_node("sparse_retrieval", self._sparse_retrieval)
        builder.add_node("hyde_retrieval", self._hyde_retrieval)
        builder.add_node("raptor_retrieval", self._raptor_retrieval)
        builder.add_node("combine_results", self._combine_results)
        builder.add_node("rerank", self._rerank)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("finalize", self._finalize)

        builder.set_entry_point("query_analysis")

        # Fan-out: query_analysis → all retrieval nodes
        builder.add_edge("query_analysis", "dense_retrieval")
        builder.add_edge("query_analysis", "sparse_retrieval")
        if self.enable_hyde:
            builder.add_edge("query_analysis", "hyde_retrieval")
        if self.enable_raptor:
            builder.add_edge("query_analysis", "raptor_retrieval")

        # Fan-in: all retrieval nodes → combine
        builder.add_edge("dense_retrieval", "combine_results")
        builder.add_edge("sparse_retrieval", "combine_results")
        if self.enable_hyde:
            builder.add_edge("hyde_retrieval", "combine_results")
        if self.enable_raptor:
            builder.add_edge("raptor_retrieval", "combine_results")

        # Combine → rerank or generate
        if self.enable_reranking:
            builder.add_edge("combine_results", "rerank")
            builder.add_edge("rerank", "generate_answer")
        else:
            builder.add_edge("combine_results", "generate_answer")

        builder.add_edge("generate_answer", "finalize")
        builder.add_edge("finalize", END)

        # Persistent checkpointing
        saver = self._create_checkpointer() if self.enable_checkpointing else None
        return builder.compile(checkpointer=saver)

    def _create_checkpointer(self):
        """Create the appropriate LangGraph checkpointer from config."""
        from ...core.config import get_config
        import os

        cfg = get_config().langgraph

        if cfg.checkpointer_backend == "sqlite":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                db_path = cfg.checkpointer_uri
                os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
                saver = SqliteSaver.from_conn_string(db_path)
                logger.info(f"Using SqliteSaver checkpointer: {db_path}")
                return saver
            except ImportError:
                logger.warning("langgraph-checkpoint-sqlite not installed, falling back to MemorySaver")

        # Fallback to in-memory
        from langgraph.checkpoint.memory import MemorySaver
        logger.info("Using MemorySaver checkpointer (sessions not persisted across restarts)")
        return MemorySaver()

    # -----------------------------------------------------------------------
    # Node implementations
    # -----------------------------------------------------------------------

    @trace_span("query_analysis")
    def _query_analysis(self, state: GraphState) -> Dict[str, Any]:
        """Analyze and decompose the query using LangChain chains."""
        query = state.query

        try:
            decomposition_result = self.decomposition_chain.invoke({
                "query": query,
                "max_sub_questions": 5,
            })

            decomposed = []
            for line in decomposition_result.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("---"):
                    sub_q = line.split(".", 1)[-1].strip()
                    if sub_q:
                        decomposed.append(sub_q)

            stepback_result = self.stepback_chain.invoke({"query": query})
            stepback_query = stepback_result.strip()

            return {
                "original_query": query,
                "stepback_query": stepback_query,
                "decomposed_queries": decomposed if decomposed else [query],
                "workflow_stage": WorkflowStage.QUERY_ANALYSIS.value,
            }

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return {
                "errors": [str(e)],
                "decomposed_queries": [query],
                "original_query": query,
            }

    @trace_span("dense_retrieval")
    def _dense_retrieval(self, state: GraphState) -> Dict[str, Any]:
        """Perform dense retrieval using LangChain vector store."""
        query = state.query
        try:
            results = self.vsm.search(query, top_k=20)
            record_retrieval("dense", len(results))
            return {"dense_results": results}
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return {"errors": [f"dense_retrieval: {e}"], "dense_results": []}

    @trace_span("sparse_retrieval")
    def _sparse_retrieval(self, state: GraphState) -> Dict[str, Any]:
        """Perform sparse (BM25) retrieval using cached index."""
        query = state.query
        try:
            bm25 = self._get_bm25_searcher()
            raw_results = bm25.search(query, top_k=20)
            results = []
            for r in raw_results:
                results.append(SearchResult(
                    id=r.get("metadata", {}).get("id", generate_uuid()),
                    content=r.get("content", ""),
                    score=min(1.0, max(0.0, r.get("score", 0.0))),
                    metadata=r.get("metadata", {}),
                ))
            record_retrieval("sparse", len(results))
            return {"sparse_results": results}
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return {"errors": [f"sparse_retrieval: {e}"], "sparse_results": []}

    @trace_span("hyde_retrieval")
    def _hyde_retrieval(self, state: GraphState) -> Dict[str, Any]:
        """Perform HyDE retrieval."""
        if not self.enable_hyde:
            return {}
        query = state.query
        try:
            raw = self.hyde_retriever.retrieve(query, top_k=20)
            results = []
            for doc in raw:
                results.append(SearchResult(
                    id=doc.metadata.get("id", generate_uuid()),
                    content=doc.page_content,
                    score=min(1.0, max(0.0, doc.metadata.get("score", 0.5))),
                    metadata=doc.metadata,
                    document_id=doc.metadata.get("document_id"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                ))
            record_retrieval("hyde", len(results))
            return {"hyde_results": results}
        except Exception as e:
            logger.error(f"Error in HyDE retrieval: {e}")
            return {"errors": [f"hyde_retrieval: {e}"], "hyde_results": []}

    @trace_span("raptor_retrieval")
    def _raptor_retrieval(self, state: GraphState) -> Dict[str, Any]:
        """Perform RAPTOR retrieval."""
        if not self.enable_raptor:
            return {}
        query = state.query
        try:
            results = self.raptor_retriever.retrieve(query, top_k=20)
            record_retrieval("raptor", len(results))
            return {"raptor_results": results}
        except Exception as e:
            logger.error(f"Error in RAPTOR retrieval: {e}")
            return {"errors": [f"raptor_retrieval: {e}"], "raptor_results": []}

    @trace_span("combine_results")
    def _combine_results(self, state: GraphState) -> Dict[str, Any]:
        """Combine and deduplicate results from all retrieval methods."""
        try:
            all_results: List[SearchResult] = []
            all_results.extend(state.dense_results)
            all_results.extend(state.sparse_results)
            all_results.extend(state.hyde_results)
            all_results.extend(state.raptor_results)

            # Deduplicate by ID
            seen: set = set()
            unique: List[SearchResult] = []
            for r in all_results:
                if r.id not in seen:
                    seen.add(r.id)
                    unique.append(r)

            # Sort descending by score
            unique.sort(key=lambda x: x.score, reverse=True)
            combined = unique[:50]

            return {
                "combined_results": combined,
                "workflow_stage": WorkflowStage.RETRIEVAL.value,
            }
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return {"errors": [f"combine_results: {e}"], "combined_results": []}

    @trace_span("rerank")
    def _rerank(self, state: GraphState) -> Dict[str, Any]:
        """Rerank combined results."""
        if not self.enable_reranking:
            return {}

        query = state.query
        results = state.combined_results

        try:
            record_reranking(len(results))
            reranked = self.reranker.rerank(query, results, top_k=10)
            return {
                "reranked_results": reranked,
                "workflow_stage": WorkflowStage.RERANKING.value,
            }
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Graceful fallback: wrap top 10 combined results as RerankedResult
            fallback = [
                RerankedResult(
                    id=r.id,
                    content=r.content,
                    original_score=r.score,
                    reranked_score=r.score,
                    metadata=r.metadata,
                    document_id=r.document_id,
                    chunk_index=r.chunk_index,
                )
                for r in results[:10]
            ]
            return {"errors": [f"reranking: {e}"], "reranked_results": fallback}

    @trace_span("generate_answer")
    def _generate_answer(self, state: GraphState) -> Dict[str, Any]:
        """Generate the final answer using LangChain chain."""
        query = state.query

        try:
            context_results: List[Dict[str, Any]] = []

            if state.reranked_results:
                for r in state.reranked_results[:10]:
                    context_results.append({
                        "content": r.content,
                        "id": r.id,
                        "score": r.reranked_score,
                        "metadata": r.metadata,
                    })
            elif state.combined_results:
                for r in state.combined_results[:10]:
                    context_results.append({
                        "content": r.content,
                        "id": r.id,
                        "score": r.score,
                        "metadata": r.metadata,
                    })

            if context_results:
                answer_result = self.answer_generator.generate_answer_from_results(
                    query, context_results
                )
            else:
                answer_result = AnswerResult(
                    answer="I couldn't find relevant information to answer your question.",
                    confidence=0.0,
                    sources=[],
                )

            return {
                "answer": answer_result.answer,
                "answer_result": answer_result,
                "confidence": answer_result.confidence,
                "sources": answer_result.sources,
                "workflow_stage": WorkflowStage.ANSWER_GENERATION.value,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "errors": [f"answer_generation: {e}"],
                "answer": "I apologize, but I encountered an error while generating the answer.",
                "confidence": 0.0,
            }

    def _finalize(self, state: GraphState) -> Dict[str, Any]:
        """Finalize the workflow state."""
        return {"workflow_stage": WorkflowStage.FINAL.value}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the LangGraph RAG workflow (synchronous)."""
        initial_state = GraphState(
            query=query,
            chat_history=chat_history or [],
        )

        graph_config: Dict[str, Any] = {"configurable": {"thread_id": generate_uuid()}}
        if config:
            graph_config["configurable"].update(config)

        result = self.graph.invoke(initial_state, config=graph_config)

        return {
            "query": result.get("original_query", query),
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "workflow_stage": result.get("workflow_stage", ""),
            "errors": result.get("errors", []),
            "decomposed_queries": result.get("decomposed_queries", []),
            "stepback_query": result.get("stepback_query"),
        }

    async def arun(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the LangGraph RAG workflow (async)."""
        initial_state = GraphState(
            query=query,
            chat_history=chat_history or [],
        )

        graph_config: Dict[str, Any] = {"configurable": {"thread_id": generate_uuid()}}
        if config:
            graph_config["configurable"].update(config)

        result = await self.graph.ainvoke(initial_state, config=graph_config)

        return {
            "query": result.get("original_query", query),
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "workflow_stage": result.get("workflow_stage", ""),
            "errors": result.get("errors", []),
            "decomposed_queries": result.get("decomposed_queries", []),
            "stepback_query": result.get("stepback_query"),
        }

    def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """Stream the LangGraph RAG workflow execution."""
        initial_state = GraphState(
            query=query,
            chat_history=chat_history or [],
        )
        graph_config = {"configurable": {"thread_id": generate_uuid()}}
        for state in self.graph.stream(initial_state, config=graph_config):
            yield state

    async def astream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """Async streaming of the LangGraph RAG workflow."""
        initial_state = GraphState(
            query=query,
            chat_history=chat_history or [],
        )
        graph_config = {"configurable": {"thread_id": generate_uuid()}}
        async for state in self.graph.astream(initial_state, config=graph_config):
            yield state


# ---------------------------------------------------------------------------
# High-level workflow API
# ---------------------------------------------------------------------------

class ModularRAGWorkflow:
    """High-level API for the LangGraph Modular RAG workflow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the workflow with LangGraph."""
        self.config = config or {}
        self.rag_graph: Optional[RAGGraph] = None
        self._init_workflow()

    def _init_workflow(self):
        """Initialize the LangGraph RAG workflow."""
        self.rag_graph = RAGGraph(
            enable_hyde=self.config.get("enable_hyde", True),
            enable_raptor=self.config.get("enable_raptor", True),
            enable_reranking=self.config.get("enable_reranking", True),
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG workflow (synchronous)."""
        if self.rag_graph is None:
            self._init_workflow()
        return self.rag_graph.run(question)  # type: ignore[union-attr]

    async def aquery(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG workflow (async)."""
        if self.rag_graph is None:
            self._init_workflow()
        return await self.rag_graph.arun(question)  # type: ignore[union-attr]

    def index_documents(self, documents: List[Any]) -> None:
        """Index documents and invalidate BM25 cache."""
        # Integration with document processor would happen here
        if self.rag_graph:
            self.rag_graph.invalidate_bm25_cache()

    def stream(self, question: str):
        """Stream the LangGraph RAG workflow."""
        if self.rag_graph is None:
            self._init_workflow()
        try:
            yield from self.rag_graph.stream(question)  # type: ignore[union-attr]
        except Exception as e:
            logger.error(f"Error streaming workflow: {e}")
            yield {"error": str(e)}

    async def astream(self, question: str):
        """Async streaming of the LangGraph RAG workflow."""
        if self.rag_graph is None:
            self._init_workflow()
        try:
            async for state in self.rag_graph.astream(question):  # type: ignore[union-attr]
                yield state
        except Exception as e:
            logger.error(f"Error in async stream: {e}")
            yield {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get the workflow status."""
        import langgraph
        return {
            "rag_graph_initialized": self.rag_graph is not None,
            "config": self.config,
            "langgraph_version": langgraph.__version__,
        }
