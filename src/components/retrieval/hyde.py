"""HyDE (Hypothetical Document Embedder) using LangChain built-in.

Replaces custom HyDERetriever with langchain.chains.HypotheticalDocumentEmbedder
which provides the full HyDE pipeline:
1. Generate hypothetical document from query using LLM
2. Embed the hypothetical document
3. Retrieve similar real documents
"""
import logging
from typing import List, Optional

from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document as LangChainDocument
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


# Default HyDE prompt template
HYDE_PROMPT = PromptTemplate.from_template(
    """Please write a passage that would answer the following question.
The passage should be detailed, factual, and directly relevant.

Question: {question}

Passage:"""
)


def create_hyde_embedder(
    llm: BaseChatModel,
    base_embeddings: Embeddings,
    prompt: Optional[PromptTemplate] = None,
) -> HypotheticalDocumentEmbedder:
    """Create a LangChain HypotheticalDocumentEmbedder.
    
    This replaces the custom HyDERetriever with a LangChain built-in
    that handles the full HyDE pipeline.
    
    Args:
        llm: LangChain chat model (e.g., ChatOllama)
        base_embeddings: Base embedding model (e.g., OllamaEmbeddings)
        prompt: Optional custom prompt template
        
    Returns:
        Configured HypotheticalDocumentEmbedder
    """
    hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=base_embeddings,
        custom_prompt=prompt or HYDE_PROMPT,
    )
    
    # Override the prompt template if custom one provided
    if prompt:
        hyde_embedder.llm_chain.prompt = prompt
    
    logger.info("Created HypotheticalDocumentEmbedder (HyDE)")
    return hyde_embedder


class HyDERetriever:
    """Backward-compatible wrapper around LangChain HypotheticalDocumentEmbedder.
    
    Provides similar interface to the original custom HyDERetriever while
    delegating to LangChain's built-in HyDE implementation.
    """
    
    def __init__(
        self,
        llm=None,
        embeddings=None,
        vector_store=None,
        num_hypothetical_docs: int = 1,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.num_hypothetical_docs = num_hypothetical_docs
        self._hyde_embedder = None
    
    def _ensure_initialized(self):
        """Lazy initialization of the HyDE embedder."""
        if self._hyde_embedder is None:
            if self.llm is None or self.embeddings is None:
                raise ValueError("LLM and embeddings must be provided")
            self._hyde_embedder = create_hyde_embedder(
                llm=self.llm,
                base_embeddings=self.embeddings,
            )
    
    def embed_query(self, query: str) -> List[float]:
        """Generate hypothetical document and embed it.
        
        This is the core HyDE step: 
        query → LLM generates hypothetical answer → embed that answer
        """
        self._ensure_initialized()
        return self._hyde_embedder.embed_query(query)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[LangChainDocument]:
        """Retrieve documents using HyDE embeddings."""
        if self.vector_store is None:
            raise ValueError("Vector store must be provided for retrieval")
        
        # Get HyDE embedding for the query
        hyde_embedding = self.embed_query(query)
        
        # Search vector store with the HyDE embedding
        results = self.vector_store.similarity_search_by_vector(
            embedding=hyde_embedding,
            k=top_k,
        )
        
        logger.info(f"HyDE retriever found {len(results)} documents")
        return results
