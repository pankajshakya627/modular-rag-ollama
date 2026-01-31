"""Vector store implementation using LangChain for Modular RAG."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import uuid

import numpy as np
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_ollama import OllamaEmbeddings

from .document_processor import Document, DocumentChunk
from ...core.embedding import get_embedding

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector stores using LangChain."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store."""
        pass


class LangChainChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store using LangChain."""
    
    def __init__(
        self,
        collection_name: str = "modular_rag_documents",
        persist_directory: str = "./data/vector_store",
        embedding_wrapper=None,
    ):
        """Initialize the LangChain ChromaDB vector store."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Get embedding function
        if embedding_wrapper is None:
            self.embedding_wrapper = get_embedding()
        else:
            self.embedding_wrapper = embedding_wrapper
        
        # Create LangChain embedding function
        self.embedding_function = self.embedding_wrapper.embedding
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize LangChain Chroma
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory=persist_directory,
        )
        
        logger.info(f"Initialized LangChain ChromaDB vector store: {collection_name}")
    
    def _doc_to_langchain(self, doc: Document) -> List[LangChainDocument]:
        """Convert Document to LangChain Document."""
        lc_docs = []
        for chunk in doc.chunks:
            metadata = {
                **chunk.metadata,
                "document_id": doc.id,
                "chunk_index": chunk.chunk_index,
                "source_type": doc.source_type,
                "source_path": doc.source_path,
            }
            lc_doc = LangChainDocument(
                page_content=chunk.content,
                metadata=metadata,
            )
            lc_docs.append(lc_doc)
        return lc_docs
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store using LangChain."""
        if not documents:
            return
        
        lc_docs = []
        for doc in documents:
            lc_docs.extend(self._doc_to_langchain(doc))
        
        # Add to LangChain Chroma
        self.vector_store.add_documents(documents=lc_docs)
        
        logger.info(f"Added {len(documents)} documents to LangChain ChromaDB")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using LangChain."""
        # Perform search using LangChain
        if filters:
            # LangChain Chroma doesn't support arbitrary filters directly
            # We'd need to use the underlying Chroma client
            results = self.vector_store.similarity_search(query, k=top_k)
        else:
            results = self.vector_store.similarity_search(query, k=top_k)
        
        search_results = []
        for doc in results:
            # Get the score (LangChain doesn't return scores by default in similarity_search)
            result = SearchResult(
                id=doc.metadata.get("id", str(uuid.uuid4())),
                content=doc.page_content,
                score=0.9,  # LangChain similarity_search doesn't return scores
                metadata=doc.metadata,
                document_id=doc.metadata.get("document_id"),
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            search_results.append(result)
        
        return search_results
    
    def search_with_score(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[LangChainDocument, float]]:
        """Search with scores using LangChain."""
        return self.vector_store.similarity_search_with_score(query, k=top_k)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        # This requires access to the underlying Chroma client
        # For now, we'll need to use the collection API
        try:
            collection = self.vector_store._collection
            for doc_id in document_ids:
                # Delete by document_id metadata
                collection.delete(
                    where={"document_id": doc_id}
                )
            logger.info(f"Deleted documents: {document_ids}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        # This would require metadata filtering
        # Simplified implementation
        results = self.vector_store.similarity_search(
            "", k=100  # Get all documents
        )
        
        for doc in results:
            if doc.metadata.get("document_id") == document_id:
                # Reconstruct document
                chunks = []
                for result in results:
                    if result.metadata.get("document_id") == document_id:
                        chunk = DocumentChunk(
                            id=result.metadata.get("id", str(uuid.uuid4())),
                            content=result.page_content,
                            metadata=result.metadata,
                            chunk_index=result.metadata.get("chunk_index", 0),
                        )
                        chunks.append(chunk)
                
                if chunks:
                    first_chunk = chunks[0]
                    document = Document(
                        id=document_id,
                        content="",  # Full content not stored
                        chunks=chunks,
                        metadata=first_chunk.metadata,
                        source_type=first_chunk.metadata.get("source_type", "text"),
                        source_path=first_chunk.metadata.get("source_path"),
                    )
                    return document
        
        return None
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        logger.info("Cleared all documents from LangChain ChromaDB")
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        try:
            return self.vector_store._collection.count()
        except:
            return 0
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the store."""
        results = self.vector_store.similarity_search("", k=10000)
        
        doc_chunks = {}
        for doc in results:
            doc_id = doc.metadata.get("document_id", "unknown")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            
            chunk = DocumentChunk(
                id=doc.metadata.get("id", str(uuid.uuid4())),
                content=doc.page_content,
                metadata=doc.metadata,
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            doc_chunks[doc_id].append(chunk)
        
        documents = []
        for doc_id, chunks in doc_chunks.items():
            chunks.sort(key=lambda x: x.chunk_index)
            
            first_chunk = chunks[0]
            document = Document(
                id=doc_id,
                content="",
                chunks=chunks,
                metadata=first_chunk.metadata,
                source_type=first_chunk.metadata.get("source_type", "text"),
                source_path=first_chunk.metadata.get("source_path"),
            )
            documents.append(document)
        
        return documents


class VectorStoreManager:
    """Manager for vector store operations with LangChain integration."""
    
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_wrapper=None,
    ):
        """Initialize the vector store manager."""
        self.vector_store = vector_store or LangChainChromaVectorStore(
            embedding_wrapper=embedding_wrapper,
        )
        self.embedding_wrapper = embedding_wrapper or get_embedding()
    
    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> None:
        """Index documents with embeddings using LangChain."""
        if not self.embedding_wrapper:
            raise ValueError("Embedding wrapper required for indexing")
        
        # Generate embeddings for all chunks
        all_contents = []
        for doc in documents:
            for chunk in doc.chunks:
                all_contents.append(chunk.content)
        
        if show_progress:
            logger.info(f"Generating embeddings for {len(all_contents)} chunks...")
        
        # Generate embeddings in batches
        embeddings = self.embedding_wrapper.embed_texts(all_contents)
        
        # Assign embeddings to chunks
        idx = 0
        for doc in documents:
            for chunk in doc.chunks:
                chunk.embedding = embeddings[idx]
                idx += 1
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        logger.info(f"Indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for documents using a text query."""
        return self.vector_store.search(query, top_k, filters)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the index."""
        self.vector_store.delete_documents(document_ids)
    
    def clear_index(self) -> None:
        """Clear the entire index."""
        self.vector_store.clear()
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        return {
            "document_count": len(self.vector_store.get_all_documents()),
            "chunk_count": self.vector_store.count(),
        }


class DenseRetriever:
    """Dense retrieval using LangChain vector store."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
    ):
        """Initialize the dense retriever."""
        self.vsm = vector_store_manager
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieve documents using dense retrieval."""
        return self.vsm.search(query, top_k, filters)
