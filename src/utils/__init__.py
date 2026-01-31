"""Utils package for Modular RAG."""
from .evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_answer_relevancy,
    compute_faithfulness,
    compute_context_precision,
)

# Note: uuid_utils is intentionally not imported here to avoid circular imports
# Import directly: from src.utils.uuid_utils import generate_uuid

__all__ = [
    'RetrievalEvaluator',
    'GenerationEvaluator',
    'RAGEvaluator',
    'compute_precision',
    'compute_recall',
    'compute_f1',
    'compute_answer_relevancy',
    'compute_faithfulness',
    'compute_context_precision',
]
