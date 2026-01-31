"""Unit tests for Modular RAG."""
import pytest
from src.core.config import Config, ConfigLoader, get_config


class TestConfig:
    """Tests for configuration module."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader singleton."""
        loader = ConfigLoader()
        assert loader is not None
    
    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert config is not None
        assert isinstance(config, Config)
    
    def test_config_sections(self):
        """Test config has required sections."""
        config = get_config()
        assert hasattr(config, 'app')
        assert hasattr(config, 'llm')
        assert hasattr(config, 'embedding')
        assert hasattr(config, 'vector_store')
        assert hasattr(config, 'hybrid_search')
        assert hasattr(config, 'reranking')
        assert hasattr(config, 'hyde')
        assert hasattr(config, 'raptor')


class TestDocumentProcessor:
    """Tests for document processor."""
    
    def test_recursive_chunker(self):
        """Test recursive chunker."""
        from src.components.retrieval.document_processor import (
            RecursiveChunker,
            DocumentChunk,
        )
        
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 20
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_fixed_size_chunker(self):
        """Test fixed size chunker."""
        from src.components.retrieval.document_processor import (
            FixedSizeChunker,
            DocumentChunk,
        )
        
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "A" * 500
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_document_creation(self):
        """Test document creation."""
        from src.components.retrieval.document_processor import Document
        
        doc = Document(
            id="test-doc",
            content="Test content",
            metadata={"source": "test"},
        )
        
        assert doc.id == "test-doc"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"


class TestSearchResult:
    """Tests for search result."""
    
    def test_search_result_creation(self):
        """Test search result creation."""
        from src.components.retrieval.vector_store import SearchResult
        
        result = SearchResult(
            id="test-id",
            content="Test content",
            score=0.95,
            metadata={"source": "test"},
        )
        
        assert result.id == "test-id"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.to_dict()["id"] == "test-id"


class TestReranking:
    """Tests for reranking components."""
    
    def test_reranked_result_creation(self):
        """Test reranked result creation."""
        from src.components.reranking.base import RerankedResult
        
        result = RerankedResult(
            id="test-id",
            content="Test content",
            original_score=0.8,
            reranked_score=0.9,
        )
        
        assert result.original_score == 0.8
        assert result.reranked_score == 0.9


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_compute_precision(self):
        """Test precision calculation."""
        from src.utils.evaluation import compute_precision
        from src.components.retrieval.vector_store import SearchResult
        
        retrieved = [
            SearchResult(id="1", content="doc1", score=0.9),
            SearchResult(id="2", content="doc2", score=0.8),
            SearchResult(id="3", content="doc3", score=0.7),
        ]
        relevant = ["1", "3"]
        
        precision = compute_precision(retrieved, relevant, k=3)
        
        assert 0 <= precision <= 1
        assert precision == 2/3
    
    def test_compute_recall(self):
        """Test recall calculation."""
        from src.utils.evaluation import compute_recall
        from src.components.retrieval.vector_store import SearchResult
        
        retrieved = [
            SearchResult(id="1", content="doc1", score=0.9),
            SearchResult(id="2", content="doc2", score=0.8),
        ]
        relevant = ["1", "2", "3"]
        
        recall = compute_recall(retrieved, relevant, k=2)
        
        assert 0 <= recall <= 1
        assert recall == 2/3
    
    def test_compute_f1(self):
        """Test F1 calculation."""
        from src.utils.evaluation import compute_f1
        
        f1 = compute_f1(0.8, 0.6)
        
        assert 0 <= f1 <= 1
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert f1 == expected


class TestAnswerResult:
    """Tests for answer generation."""
    
    def test_answer_result_creation(self):
        """Test answer result creation."""
        from src.components.generation.answer_generator import AnswerResult
        
        result = AnswerResult(
            answer="Test answer",
            sources=[{"id": "1", "content": "source"}],
            confidence=0.85,
        )
        
        assert result.answer == "Test answer"
        assert len(result.sources) == 1
        assert result.confidence == 0.85


class TestAPIModels:
    """Tests for API models."""
    
    def test_query_request(self):
        """Test query request model."""
        from src.api.models import QueryRequest
        
        request = QueryRequest(query="Test query")
        
        assert request.query == "Test query"
        assert request.use_hyde is True
    
    def test_health_response(self):
        """Test health response model."""
        from src.api.models import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            llm_available=True,
            embedding_available=True,
        )
        
        assert response.status == "healthy"
        assert response.llm_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
