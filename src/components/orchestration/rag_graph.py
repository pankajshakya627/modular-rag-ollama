"""LangGraph orchestration for Modular RAG workflow."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
import logging
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..retrieval.vector_store import SearchResult
from ..retrieval.hyde import HyDERetriever
from ..retrieval.hybrid_search import HybridSearcher
from ..retrieval.raptor import RAPTORRetriever
from ..reranking.base import BaseReranker, RerankedResult
from ..generation.answer_generator import AnswerGenerator, AnswerResult
from ...core.llm import LLMWrapper, get_llm
from ...core.embedding import EmbeddingWrapper, get_embedding
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


class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    # Input
    query: str
    chat_history: List[Dict[str, str]]
    
    # Query Processing
    original_query: str
    decomposed_queries: List[str]
    stepback_query: Optional[str]
    hyde_query: Optional[str]
    
    # Retrieval Results
    dense_results: List[SearchResult]
    sparse_results: List[SearchResult]
    hyde_results: List[SearchResult]
    raptor_results: List[SearchResult]
    combined_results: List[SearchResult]
    
    # Reranked Results
    reranked_results: List[RerankedResult]
    
    # Generation
    answer: str
    answer_result: AnswerResult
    
    # Metadata
    workflow_stage: str
    errors: List[str]
    confidence: float
    sources: List[Dict[str, Any]]


class RAGGraph:
    """LangGraph-based RAG workflow with modular components."""
    
    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        embedding_wrapper: Optional[EmbeddingWrapper] = None,
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
        
        # Initialize components
        self._init_components()
        
        # Build the LangGraph
        self.graph = self._build_graph()
    
    def _init_components(self):
        """Initialize RAG components using LangChain."""
        from ...core.config import get_config
        from ..retrieval.vector_store import LangChainChromaVectorStore, VectorStoreManager
        
        config = get_config()
        
        # Vector store manager with LangChain Chroma
        self.vector_store = LangChainChromaVectorStore(
            collection_name=config.vector_store.collection_name,
            persist_directory=config.vector_store.persist_directory,
            embedding_wrapper=self.embedding_wrapper,
        )
        self.vsm = VectorStoreManager(
            vector_store=self.vector_store,
            embedding_wrapper=self.embedding_wrapper,
        )
        
        # Hybrid search
        self.hybrid_searcher = HybridSearcher(
            vector_store_manager=self.vsm,
            alpha=config.hybrid_search.alpha,
            top_k=config.hybrid_search.top_k,
        )
        
        # HyDE retriever
        self.hyde_retriever = HyDERetriever(
            vector_store_manager=self.vsm,
            llm_wrapper=self.llm_wrapper,
        )
        
        # RAPTOR retriever
        self.raptor_retriever = RAPTORRetriever(
            vector_store_manager=self.vsm,
            llm_wrapper=self.llm_wrapper,
            embedding_wrapper=self.embedding_wrapper,
            num_clusters=config.raptor.num_clusters,
        )
        
        # Reranker (using LLM-based)
        from ..reranking.base import LLMEnsembleReranker
        self.reranker = LLMEnsembleReranker(llm_wrapper=self.llm_wrapper)
        
        # Answer generator with LangChain
        self.answer_generator = AnswerGenerator(llm_wrapper=self.llm_wrapper)
        
        # Create LangChain chains for query processing
        self._create_query_chains()
    
    def _create_query_chains(self):
        """Create LangChain chains for query processing using LCEL."""
        # Output parser for string output
        self.output_parser = StrOutputParser()
        
        # Decomposition chain using LCEL pattern
        self.decomposition_prompt = PromptTemplate(
            template="""Decompose the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.
Return exactly {max_sub_questions} sub-questions.

Original Question: {query}

Sub-questions (one per line):""",
            input_variables=["query", "max_sub_questions"],
        )
        # LCEL chain: prompt | llm | parser
        self.decomposition_chain = self.decomposition_prompt | self.llm_wrapper.llm | self.output_parser
        
        # Step-back chain using LCEL pattern
        self.stepback_prompt = PromptTemplate(
            template="""Generate a broader, more general question that captures the context of the original query.
This helps retrieve relevant background information.

Original Question: {query}

Step-back Question:""",
            input_variables=["query"],
        )
        self.stepback_chain = self.stepback_prompt | self.llm_wrapper.llm | self.output_parser
        
        # HyDE chain using LCEL pattern
        self.hyde_prompt = PromptTemplate(
            template="""Write a detailed hypothetical answer to the question.
Include specific facts, examples, and reasoning that would appear in a relevant document.

Question: {query}

Hypothetical Answer:""",
            input_variables=["query"],
        )
        self.hyde_chain = self.hyde_prompt | self.llm_wrapper.llm | self.output_parser
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the LangGraph state machine
        builder = StateGraph(GraphState)
        
        # Add nodes using LangGraph
        builder.add_node("query_analysis", self._query_analysis)
        builder.add_node("dense_retrieval", self._dense_retrieval)
        builder.add_node("sparse_retrieval", self._sparse_retrieval)
        builder.add_node("hyde_retrieval", self._hyde_retrieval)
        builder.add_node("raptor_retrieval", self._raptor_retrieval)
        builder.add_node("combine_results", self._combine_results)
        builder.add_node("rerank", self._rerank)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("finalize", self._finalize)
        
        # Set entry point
        builder.set_entry_point("query_analysis")
        
        # Add edges with conditional routing
        builder.add_edge("query_analysis", "dense_retrieval")
        builder.add_edge("query_analysis", "sparse_retrieval")
        
        if self.enable_hyde:
            builder.add_edge("query_analysis", "hyde_retrieval")
        
        if self.enable_raptor:
            builder.add_edge("query_analysis", "raptor_retrieval")
        
        # All retrieval methods -> combine
        builder.add_edge("dense_retrieval", "combine_results")
        builder.add_edge("sparse_retrieval", "combine_results")
        
        if self.enable_hyde:
            builder.add_edge("hyde_retrieval", "combine_results")
        
        if self.enable_raptor:
            builder.add_edge("raptor_retrieval", "combine_results")
        
        # Combine -> rerank or generate
        if self.enable_reranking:
            builder.add_edge("combine_results", "rerank")
        else:
            builder.add_edge("combine_results", "generate_answer")
        
        # Rerank -> generate
        if self.enable_reranking:
            builder.add_edge("rerank", "generate_answer")
        
        # Generate -> finalize
        builder.add_edge("generate_answer", "finalize")
        
        # Finalize -> END
        builder.add_edge("finalize", END)
        
        # Create checkpoint saver for LangGraph
        saver = MemorySaver() if self.enable_checkpointing else None
        
        return builder.compile(checkpointer=saver)
    
    def _query_analysis(self, state: GraphState) -> GraphState:
        """Analyze and decompose the query using LangChain chains."""
        query = state["query"]
        
        try:
            # Use LCEL chain for decomposition
            decomposition_result = self.decomposition_chain.invoke({
                "query": query,
                "max_sub_questions": 5,
            })
            
            # Parse sub-questions (LCEL returns string directly)
            decomposed = []
            for line in decomposition_result.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('---'):
                    sub_q = line.split('.', 1)[-1].strip()
                    if sub_q:
                        decomposed.append(sub_q)
            
            # Use LCEL chain for stepback
            stepback_result = self.stepback_chain.invoke({"query": query})
            stepback_query = stepback_result.strip()
            
            state["original_query"] = query
            state["stepback_query"] = stepback_query
            state["decomposed_queries"] = decomposed if decomposed else [query]
            state["workflow_stage"] = WorkflowStage.QUERY_ANALYSIS.value
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            state["errors"] = state.get("errors", []) + [str(e)]
            state["decomposed_queries"] = [query]
        
        return state
    
    def _dense_retrieval(self, state: GraphState) -> GraphState:
        """Perform dense retrieval using LangChain vector store."""
        query = state["query"]
        
        try:
            results = self.vsm.search(query, top_k=20)
            state["dense_results"] = results
            state["workflow_stage"] = WorkflowStage.RETRIEVAL.value
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            state["errors"] = state.get("errors", []) + [f"dense_retrieval: {str(e)}"]
            state["dense_results"] = []
        
        return state
    
    def _sparse_retrieval(self, state: GraphState) -> GraphState:
        """Perform sparse (BM25) retrieval."""
        query = state["query"]
        
        try:
            from ..retrieval.hybrid_search import BM25Searcher
            bm25 = BM25Searcher()
            all_docs = self.vsm.vector_store.get_all_documents()
            if all_docs:
                bm25.build_index(all_docs)
                results = bm25.search(query, top_k=20)
                state["sparse_results"] = results
            else:
                state["sparse_results"] = []
            state["workflow_stage"] = WorkflowStage.RETRIEVAL.value
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            state["errors"] = state.get("errors", []) + [f"sparse_retrieval: {str(e)}"]
            state["sparse_results"] = []
        
        return state
    
    def _hyde_retrieval(self, state: GraphState) -> GraphState:
        """Perform HyDE retrieval using LangChain chain."""
        if not self.enable_hyde:
            return state
        
        query = state["query"]
        
        try:
            # Generate hypothetical document using LCEL chain
            hyde_result = self.hyde_chain.invoke({"query": query})
            hyde_query = hyde_result.strip()
            
            # Use the hypothetical document for retrieval
            results = self.hyde_retriever.retrieve(query, top_k=20)
            state["hyde_results"] = results
        except Exception as e:
            logger.error(f"Error in HyDE retrieval: {e}")
            state["errors"] = state.get("errors", []) + [f"hyde_retrieval: {str(e)}"]
            state["hyde_results"] = []
        
        return state
    
    def _raptor_retrieval(self, state: GraphState) -> GraphState:
        """Perform RAPTOR retrieval."""
        if not self.enable_raptor:
            return state
        
        query = state["query"]
        
        try:
            results = self.raptor_retriever.retrieve(query, top_k=20)
            state["raptor_results"] = results
        except Exception as e:
            logger.error(f"Error in RAPTOR retrieval: {e}")
            state["errors"] = state.get("errors", []) + [f"raptor_retrieval: {str(e)}"]
            state["raptor_results"] = []
        
        return state
    
    def _combine_results(self, state: GraphState) -> GraphState:
        """Combine results from all retrieval methods."""
        try:
            all_results = []
            
            # Add dense results
            for r in state.get("dense_results", []):
                all_results.append({**r.to_dict(), "source": "dense"})
            
            # Add sparse results
            for r in state.get("sparse_results", []):
                all_results.append({**r.to_dict(), "source": "sparse"})
            
            # Add HyDE results
            for r in state.get("hyde_results", []):
                all_results.append({**r.to_dict(), "source": "hyde"})
            
            # Add RAPTOR results
            for r in state.get("raptor_results", []):
                all_results.append({**r.to_dict(), "source": "raptor"})
            
            # Deduplicate by ID
            seen = set()
            unique_results = []
            for r in all_results:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    unique_results.append(r)
            
            # Sort by score
            unique_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Convert back to SearchResult objects
            combined = []
            for r in unique_results[:50]:
                combined.append(SearchResult(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                    document_id=r.get("document_id"),
                    chunk_index=r.get("chunk_index", 0),
                ))
            
            state["combined_results"] = combined
            state["workflow_stage"] = WorkflowStage.RETRIEVAL.value
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            state["errors"] = state.get("errors", []) + [f"combine_results: {str(e)}"]
            state["combined_results"] = []
        
        return state
    
    def _rerank(self, state: GraphState) -> GraphState:
        """Rerank combined results."""
        if not self.enable_reranking:
            return state
        
        query = state["query"]
        results = state.get("combined_results", [])
        
        try:
            reranked = self.reranker.rerank(query, results, top_k=10)
            state["reranked_results"] = reranked
            state["workflow_stage"] = WorkflowStage.RERANKING.value
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            state["errors"] = state.get("errors", []) + [f"reranking: {str(e)}"]
            state["reranked_results"] = results[:10]
        
        return state
    
    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate the final answer."""
        query = state["query"]
        
        try:
            # Use reranked results if available, else combined results
            results = state.get("reranked_results", state.get("combined_results", []))
            
            if results:
                answer_result = self.answer_generator.generate_answer_from_results(
                    query, results[:10]
                )
                state["answer"] = answer_result.answer
                state["answer_result"] = answer_result
                state["confidence"] = answer_result.confidence
                state["sources"] = answer_result.sources
            else:
                state["answer"] = "I couldn't find relevant information to answer your question."
                state["confidence"] = 0.0
                state["sources"] = []
            
            state["workflow_stage"] = WorkflowStage.ANSWER_GENERATION.value
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["errors"] = state.get("errors", []) + [f"answer_generation: {str(e)}"]
            state["answer"] = "I apologize, but I encountered an error while generating the answer."
            state["confidence"] = 0.0
        
        return state
    
    def _finalize(self, state: GraphState) -> GraphState:
        """Finalize the workflow state."""
        state["workflow_stage"] = WorkflowStage.FINAL.value
        return state
    
    def run(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the LangGraph RAG workflow."""
        initial_state: GraphState = {
            "query": query,
            "chat_history": chat_history or [],
            "original_query": query,
            "decomposed_queries": [],
            "stepback_query": None,
            "hyde_query": None,
            "dense_results": [],
            "sparse_results": [],
            "hyde_results": [],
            "raptor_results": [],
            "combined_results": [],
            "reranked_results": [],
            "answer": "",
            "answer_result": None,
            "workflow_stage": "",
            "errors": [],
            "confidence": 0.0,
            "sources": [],
        }
        
        # LangGraph config for checkpointing
        graph_config = {"configurable": {"thread_id": generate_uuid()}}
        if config:
            graph_config["configurable"].update(config)
        
        # Invoke the LangGraph
        result = self.graph.invoke(initial_state, config=graph_config)
        
        return {
            "query": result["original_query"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "workflow_stage": result["workflow_stage"],
            "errors": result.get("errors", []),
            "decomposed_queries": result.get("decomposed_queries", []),
            "stepback_query": result.get("stepback_query"),
        }
    
    def stream(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """Stream the LangGraph RAG workflow execution."""
        initial_state: GraphState = {
            "query": query,
            "chat_history": chat_history or [],
            "original_query": query,
            "decomposed_queries": [],
            "stepback_query": None,
            "hyde_query": None,
            "dense_results": [],
            "sparse_results": [],
            "hyde_results": [],
            "raptor_results": [],
            "combined_results": [],
            "reranked_results": [],
            "answer": "",
            "answer_result": None,
            "workflow_stage": "",
            "errors": [],
            "confidence": 0.0,
            "sources": [],
        }
        
        # Stream the LangGraph
        config = {"configurable": {"thread_id": generate_uuid()}}
        for state in self.graph.stream(initial_state, config=config):
            yield state


class ModularRAGWorkflow:
    """High-level API for the LangGraph Modular RAG workflow."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the workflow with LangGraph."""
        self.config = config or {}
        self.rag_graph = None
        
        # Initialize LangGraph components
        self._init_workflow()
    
    def _init_workflow(self):
        """Initialize the LangGraph RAG workflow."""
        self.rag_graph = RAGGraph(
            enable_hyde=self.config.get("enable_hyde", True),
            enable_raptor=self.config.get("enable_raptor", True),
            enable_reranking=self.config.get("enable_reranking", True),
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the LangGraph RAG workflow."""
        if self.rag_graph is None:
            self._init_workflow()
        
        return self.rag_graph.run(question)
    
    def index_documents(self, documents: List[Any]) -> None:
        """Index documents for retrieval using LangChain."""
        pass  # Would need to integrate with document processor
    
    def get_status(self) -> Dict[str, Any]:
        """Get the workflow status."""
        return {
            "rag_graph_initialized": self.rag_graph is not None,
            "config": self.config,
            "langgraph_version": "0.1.0",
        }
