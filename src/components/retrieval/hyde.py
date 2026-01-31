"""HyDE (Hypothetical Document Embeddings) implementation."""
from typing import Any, Dict, List, Optional
import logging

from .document_processor import Document, DocumentChunk
from .vector_store import SearchResult, VectorStoreManager

logger = logging.getLogger(__name__)


class HyDERetriever:
    """HyDE retriever that uses hypothetical document embeddings for retrieval."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_wrapper,
        hypothesis_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        """Initialize the HyDE retriever."""
        self.vsm = vector_store_manager
        self.llm_wrapper = llm_wrapper
        self.hypothesis_prompt = hypothesis_prompt or """Write a detailed hypothetical answer to the question. 
Include specific facts, examples, and reasoning that would appear in a relevant document.

Question: {query}

Hypothetical Answer:"""
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document for the query."""
        prompt = self.hypothesis_prompt.format(query=query)
        
        try:
            response = self.llm_wrapper.llm.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            # Fallback: return the query itself
            return query
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_hyde: bool = True,
    ) -> List[SearchResult]:
        """Retrieve documents using HyDE."""
        if use_hyde:
            # Generate hypothetical document
            logger.info("Generating hypothetical document for query...")
            hypothetical_doc = self.generate_hypothetical_document(query)
            
            # Use the hypothetical document for retrieval
            results = self.vsm.search(
                query=hypothetical_doc,
                top_k=top_k,
                filters=filters,
            )
            
            logger.info(f"HyDE retrieval: found {len(results)} results using hypothetical document")
            return results
        else:
            # Standard retrieval
            return self.vsm.search(query, top_k, filters)
    
    def retrieve_with_comparison(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[SearchResult]]:
        """Retrieve documents using both HyDE and standard retrieval for comparison."""
        # Standard retrieval
        standard_results = self.vsm.search(query, top_k, filters)
        
        # HyDE retrieval
        hyde_results = self.retrieve(query, top_k, filters, use_hyde=True)
        
        return {
            "standard": standard_results,
            "hyde": hyde_results,
        }
    
    def set_hypothesis_prompt(self, prompt: str) -> None:
        """Set a custom hypothesis prompt."""
        self.hypothesis_prompt = prompt
    
    def set_llm(self, llm_wrapper) -> None:
        """Set a different LLM for hypothetical document generation."""
        self.llm_wrapper = llm_wrapper
