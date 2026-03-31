"""Unit and integration tests for Modular RAG.

Coverage:
- Config validation (field_validator, model_validator)
- Pydantic model validation (SearchResult, DocumentChunk, Document, RerankedResult)
- API model validation (field_validator on query, chunking_strategy)
- HTTP API integration using httpx.AsyncClient
- Answer generation (with mock LLM)
"""
import pytest
import pytest_asyncio
from pydantic import ValidationError

from src.core.config import Config, get_config, LLMConfig, CacheConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for configuration module."""

    def test_config_loader_initialization(self):
        from src.core.config import ConfigLoader
        loader = ConfigLoader()
        assert loader is not None

    def test_get_config(self):
        config = get_config()
        assert config is not None
        assert isinstance(config, Config)

    def test_config_sections(self):
        config = get_config()
        assert hasattr(config, "app")
        assert hasattr(config, "llm")
        assert hasattr(config, "embedding")
        assert hasattr(config, "vector_store")
        assert hasattr(config, "hybrid_search")
        assert hasattr(config, "reranking")
        assert hasattr(config, "hyde")
        assert hasattr(config, "raptor")
        assert hasattr(config, "langgraph")

    def test_llm_config_temperature_bounds(self):
        """Temperature must be in [0.0, 2.0]."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=3.0)
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)
        # Valid
        cfg = LLMConfig(temperature=1.5)
        assert cfg.temperature == 1.5

    def test_cache_config_yaml_alias(self):
        """CacheConfig accepts 'type' as alias for 'cache_type'."""
        # Simulates loading from YAML where key is 'type'
        cfg = CacheConfig(**{"type": "memory", "max_size": 500})
        assert cfg.cache_type == "memory"

    def test_config_is_frozen(self):
        """Config objects must be immutable after construction."""
        config = get_config()
        with pytest.raises(Exception):  # ValidationError or TypeError
            config.app.debug = True  # type: ignore[misc]

    def test_document_processing_overlap_validation(self):
        """chunk_overlap must be less than chunk_size."""
        from src.core.config import DocumentProcessingConfig
        with pytest.raises(ValidationError):
            DocumentProcessingConfig(chunk_size=256, chunk_overlap=256)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentChunk:
    """Tests for DocumentChunk Pydantic model."""

    def test_creation_with_defaults(self, sample_chunk):
        assert sample_chunk.id == "chunk-test-001"
        assert sample_chunk.content == "This is a sample chunk for testing purposes."
        assert sample_chunk.embedding is None
        assert sample_chunk.chunk_index == 0

    def test_auto_id_generation(self):
        from src.components.retrieval.document_processor import DocumentChunk
        chunk = DocumentChunk(content="auto id test")
        assert chunk.id  # non-empty string
        assert len(chunk.id) > 0

    def test_to_dict_excludes_embedding(self, sample_chunk):
        d = sample_chunk.to_dict()
        assert "id" in d
        assert "content" in d
        # embedding excluded from to_dict
        assert "embedding" not in d

    def test_chunk_index_non_negative(self):
        from src.components.retrieval.document_processor import DocumentChunk
        with pytest.raises(ValidationError):
            DocumentChunk(content="x", chunk_index=-1)


class TestDocument:
    """Tests for Document Pydantic model."""

    def test_creation(self, sample_document):
        assert sample_document.id == "doc-test-001"
        assert sample_document.source_type == ".txt"
        assert len(sample_document.chunks) == 1

    def test_auto_id_generation(self):
        from src.components.retrieval.document_processor import Document
        doc = Document(content="test")
        assert doc.id

    def test_from_dict_roundtrip(self, sample_document):
        d = sample_document.to_dict()
        restored = sample_document.from_dict(d)
        assert restored.id == sample_document.id


class TestSearchResult:
    """Tests for SearchResult Pydantic model."""

    def test_creation(self, sample_search_results):
        r = sample_search_results[0]
        assert r.id == "chunk-001"
        assert r.score == 0.92
        assert r.chunk_index == 0

    def test_score_out_of_range(self):
        from src.components.retrieval.vector_store import SearchResult
        with pytest.raises(ValidationError):
            SearchResult(id="x", content="x", score=1.5)
        with pytest.raises(ValidationError):
            SearchResult(id="x", content="x", score=-0.1)

    def test_to_dict(self, sample_search_results):
        d = sample_search_results[0].to_dict()
        assert d["id"] == "chunk-001"
        assert d["score"] == 0.92

    def test_negative_chunk_index_rejected(self):
        from src.components.retrieval.vector_store import SearchResult
        with pytest.raises(ValidationError):
            SearchResult(id="x", content="x", score=0.5, chunk_index=-1)


class TestRerankedResult:
    """Tests for RerankedResult Pydantic model."""

    def test_creation(self, sample_reranked_results):
        r = sample_reranked_results[0]
        assert r.original_score == 0.92
        assert r.reranked_score == 0.95
        assert isinstance(r.metadata, dict)

    def test_metadata_defaults_to_empty_dict(self):
        from src.components.reranking.base import RerankedResult
        r = RerankedResult(
            id="x", content="x", original_score=0.5, reranked_score=0.6
        )
        assert r.metadata == {}

    def test_to_dict_uses_model_dump(self, sample_reranked_results):
        d = sample_reranked_results[0].to_dict()
        assert "reranked_score" in d
        assert "original_score" in d


class TestAnswerResult:
    """Tests for AnswerResult Pydantic model."""

    def test_creation(self):
        from src.components.generation.answer_generator import AnswerResult
        result = AnswerResult(
            answer="Revenue grew 12% in Q3.",
            sources=[{"id": "chunk-001", "content": "..."}],
            confidence=0.85,
        )
        assert result.answer.startswith("Revenue")
        assert len(result.sources) == 1
        assert result.confidence == 0.85

    def test_confidence_bounds(self):
        from src.components.generation.answer_generator import AnswerResult
        with pytest.raises(ValidationError):
            AnswerResult(answer="x", confidence=1.5)
        with pytest.raises(ValidationError):
            AnswerResult(answer="x", confidence=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Document Processor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentProcessor:
    """Tests for DocumentProcessor with LangChain splitters."""

    def test_recursive_chunker(self):
        from src.components.retrieval.document_processor import RecursiveChunker, DocumentChunk
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("This is a test sentence. " * 20)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_fixed_size_chunker(self):
        from src.components.retrieval.document_processor import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("A" * 500)
        assert len(chunks) > 0

    def test_process_text(self):
        from src.components.retrieval.document_processor import DocumentProcessor, Document
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        doc = processor.process_text("Hello world. " * 30)
        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0
        assert doc.source_type == "text"

    def test_clean_text_preserves_unicode(self):
        from src.components.retrieval.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        text = "Hello 世界 مرحبا"
        cleaned = processor._clean_text(text)
        assert "世界" in cleaned
        assert "مرحبا" in cleaned


# ─────────────────────────────────────────────────────────────────────────────
# API Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIModels:
    """Tests for API request/response models."""

    def test_query_request_strips_whitespace(self):
        from src.api.models import QueryRequest
        req = QueryRequest(query="  What is revenue?  ")
        assert req.query == "What is revenue?"

    def test_query_request_empty_raises(self):
        from src.api.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_query_request_too_short_raises(self):
        from src.api.models import QueryRequest
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_chunking_strategy_validation(self):
        from src.api.models import IndexRequest
        with pytest.raises(ValidationError):
            IndexRequest(text_content="x", chunking_strategy="unknown_strategy")
        # Valid strategies
        for strategy in ("recursive", "semantic", "fixed_size", "sentence"):
            req = IndexRequest(text_content="x", chunking_strategy=strategy)
            assert req.chunking_strategy == strategy

    def test_index_request_overlap_validation(self):
        from src.api.models import IndexRequest
        with pytest.raises(ValidationError):
            IndexRequest(text_content="x", chunk_size=100, chunk_overlap=100)

    def test_health_response(self):
        from src.api.models import HealthResponse
        r = HealthResponse(
            status="healthy",
            version="1.0.0",
            llm_available=True,
            embedding_available=True,
        )
        assert r.status == "healthy"
        assert r.llm_available is True


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Integration Tests (requires running app — skipped if Ollama unavailable)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests using httpx.AsyncClient with test app."""

    async def test_health_endpoint(self):
        """Health endpoint returns 200 even when LLM is down."""
        import httpx
        from src.api.main import app

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    async def test_query_endpoint_validation_error(self):
        """Empty query should return 422 Unprocessable Entity."""
        import httpx
        from src.api.main import app

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/query", json={"query": ""})
        assert response.status_code == 422

    async def test_metrics_endpoint(self):
        """Metrics endpoint returns Prometheus text."""
        import httpx
        from src.api.main import app

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Observability Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestObservability:
    """Tests for observability module."""

    def test_setup_logging_does_not_raise(self):
        from src.core.observability import setup_logging
        setup_logging("INFO")  # Should not raise

    def test_get_logger(self):
        from src.core.observability import get_logger
        log = get_logger("test.module")
        assert log is not None

    def test_trace_span_decorator(self):
        from src.core.observability import trace_span

        @trace_span("test_function")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_trace_span_async_decorator(self):
        from src.core.observability import trace_span

        @trace_span("test_async_function")
        async def my_async_func(x: int) -> int:
            return x * 3

        result = await my_async_func(4)
        assert result == 12

    def test_record_metrics_no_crash(self):
        from src.core.observability import (
            record_query_duration,
            record_query_error,
            record_retrieval,
            record_reranking,
        )
        # Should not raise even if prometheus_client not installed
        record_query_duration(0.5)
        record_query_error("test_stage")
        record_retrieval("dense", 10)
        record_reranking(5)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Tests (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    def test_compute_precision(self):
        from src.utils.evaluation import compute_precision
        from src.components.retrieval.vector_store import SearchResult

        retrieved = [
            SearchResult(id="1", content="doc1", score=0.9),
            SearchResult(id="2", content="doc2", score=0.8),
            SearchResult(id="3", content="doc3", score=0.7),
        ]
        relevant = ["1", "3"]
        precision = compute_precision(retrieved, relevant, k=3)
        assert precision == pytest.approx(2 / 3)

    def test_compute_recall(self):
        from src.utils.evaluation import compute_recall
        from src.components.retrieval.vector_store import SearchResult

        retrieved = [
            SearchResult(id="1", content="doc1", score=0.9),
            SearchResult(id="2", content="doc2", score=0.8),
        ]
        relevant = ["1", "2", "3"]
        recall = compute_recall(retrieved, relevant, k=2)
        assert recall == pytest.approx(2 / 3)

    def test_compute_f1(self):
        from src.utils.evaluation import compute_f1

        f1 = compute_f1(0.8, 0.6)
        expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        assert f1 == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
