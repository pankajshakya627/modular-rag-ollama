"""HyDE (Hypothetical Document Embedder) using LangChain LCEL.

Provides a production-grade HyDE pipeline:
1. Generate hypothetical document from query using LLM
2. Embed the hypothetical document
3. Retrieve similar real documents via vector similarity
"""
import logging
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
) -> "HyDEEmbedder":
    """Create a HyDE embedder using LCEL.

    Generates a hypothetical document from the query via LLM,
    then embeds it for vector similarity search.

    Args:
        llm: LangChain chat model (e.g., ChatOllama)
        base_embeddings: Base embedding model (e.g., OllamaEmbeddings)
        prompt: Optional custom prompt template

    Returns:
        Configured HyDEEmbedder instance
    """
    embedder = HyDEEmbedder(
        llm=llm,
        base_embeddings=base_embeddings,
        prompt=prompt,
    )
    logger.info("Created LCEL-based HyDE embedder")
    return embedder


class HyDEEmbedder:
    """LCEL-based HyDE embedder: query -> LLM hypothesis -> embedding."""

    def __init__(
        self,
        llm: BaseChatModel,
        base_embeddings: Embeddings,
        prompt: Optional[PromptTemplate] = None,
    ):
        self.llm = llm
        self.base_embeddings = base_embeddings
        _prompt = prompt or HYDE_PROMPT
        self._chain = _prompt | llm | StrOutputParser()

    def embed_query(self, query: str) -> List[float]:
        """Generate hypothetical document and embed it."""
        hypothesis = self._chain.invoke({"question": query})
        return self.base_embeddings.embed_query(hypothesis)

    async def aembed_query(self, query: str) -> List[float]:
        """Async version of embed_query."""
        hypothesis = await self._chain.ainvoke({"question": query})
        return await self.base_embeddings.aembed_query(hypothesis)


class HyDERetriever:
    """Backward-compatible HyDE retriever using LCEL-based HyDEEmbedder.

    Provides the same interface as the original custom HyDERetriever while
    using a reliable LCEL implementation that avoids deprecated LangChain chains.
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
        self._hyde_embedder: Optional[HyDEEmbedder] = None

    def _ensure_initialized(self):
        """Lazy initialization of the HyDE embedder."""
        if self._hyde_embedder is None:
            if self.llm is None or self.embeddings is None:
                raise ValueError("LLM and embeddings must be provided")
            self._hyde_embedder = HyDEEmbedder(
                llm=self.llm,
                base_embeddings=self.embeddings,
            )

    def embed_query(self, query: str) -> List[float]:
        """Generate hypothetical document and embed it."""
        self._ensure_initialized()
        return self._hyde_embedder.embed_query(query)

    async def aembed_query(self, query: str) -> List[float]:
        """Async version of embed_query."""
        self._ensure_initialized()
        return await self._hyde_embedder.aembed_query(query)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[LangChainDocument]:
        """Retrieve documents using HyDE embeddings."""
        if self.vector_store is None:
            raise ValueError("Vector store must be provided for retrieval")

        hyde_embedding = self.embed_query(query)

        results = self.vector_store.similarity_search_by_vector(
            embedding=hyde_embedding,
            k=top_k,
        )

        logger.info(f"HyDE retriever found {len(results)} documents")
        return results

    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[LangChainDocument]:
        """Async version of retrieve."""
        if self.vector_store is None:
            raise ValueError("Vector store must be provided for retrieval")

        hyde_embedding = await self.aembed_query(query)

        # Chroma's async search (falls back to sync if not supported)
        try:
            results = await self.vector_store.asimilarity_search_by_vector(
                embedding=hyde_embedding,
                k=top_k,
            )
        except AttributeError:
            results = self.vector_store.similarity_search_by_vector(
                embedding=hyde_embedding,
                k=top_k,
            )

        logger.info(f"HyDE async retriever found {len(results)} documents")
        return results
