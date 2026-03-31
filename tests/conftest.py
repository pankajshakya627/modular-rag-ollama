"""pytest fixtures for Modular RAG tests.

Provides:
- mock_llm: FakeListChatModel — no Ollama required
- mock_embedder: MagicMock with proper embed_* methods
- sample_search_results: Pre-built SearchResult list
- sample_documents: Pre-built Document list for indexing tests
"""
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import GenericFakeChatModel

from src.components.retrieval.vector_store import SearchResult
from src.components.retrieval.document_processor import Document, DocumentChunk
from src.components.reranking.base import RerankedResult


# ─────────────────────────────────────────────────────────────────────────────
# LLM Mocks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Returns a GenericFakeChatModel that returns canned AI messages."""
    from langchain_core.messages import AIMessage
    return GenericFakeChatModel(
        messages=iter([
            AIMessage(content="This is a mocked answer about revenue in Q3 2024."),
            AIMessage(content="Sub-question 1\nSub-question 2\nSub-question 3"),
            AIMessage(content="Broader general question about financial performance"),
            AIMessage(content="0.85"),
        ])
    )


@pytest.fixture
def mock_embedder():
    """Returns a mock OllamaEmbeddings with deterministic embeddings."""
    m = MagicMock()
    m.embed_query.return_value = [0.1] * 768
    m.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
    m.aembed_query = AsyncMock(return_value=[0.1] * 768)
    m.aembed_documents = AsyncMock(return_value=[[0.1] * 768])
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Pre-built SearchResult objects for retrieval tests."""
    return [
        SearchResult(
            id="chunk-001",
            content="Q3 2024 revenue was $5.2 billion, up 12% year-over-year.",
            score=0.92,
            metadata={"source": "earnings_report.pdf", "quarter": "Q3"},
            document_id="doc-001",
            chunk_index=0,
        ),
        SearchResult(
            id="chunk-002",
            content="Operating income for Q3 2024 was $1.1 billion.",
            score=0.85,
            metadata={"source": "earnings_report.pdf", "quarter": "Q3"},
            document_id="doc-001",
            chunk_index=1,
        ),
        SearchResult(
            id="chunk-003",
            content="The company expects Q4 revenue to grow 8–10%.",
            score=0.72,
            metadata={"source": "earnings_report.pdf", "quarter": "Q4"},
            document_id="doc-001",
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_reranked_results() -> List[RerankedResult]:
    """Pre-built RerankedResult objects for reranking tests."""
    return [
        RerankedResult(
            id="chunk-001",
            content="Q3 2024 revenue was $5.2 billion, up 12% year-over-year.",
            original_score=0.92,
            reranked_score=0.95,
            metadata={"source": "earnings_report.pdf"},
            document_id="doc-001",
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """A single DocumentChunk for unit testing."""
    return DocumentChunk(
        id="chunk-test-001",
        content="This is a sample chunk for testing purposes.",
        metadata={"page": 1, "section": "intro"},
        chunk_index=0,
    )


@pytest.fixture
def sample_document(sample_chunk) -> Document:
    """A Document containing one chunk."""
    return Document(
        id="doc-test-001",
        content="This is a sample document for testing purposes.",
        chunks=[sample_chunk],
        metadata={"author": "Test Author"},
        source_type=".txt",
        source_path="/tmp/test.txt",
    )
