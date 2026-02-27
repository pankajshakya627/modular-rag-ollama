"""Core models for Modular RAG using LangChain."""
from .config import get_config, Config, ConfigLoader
from .llm import get_llm, create_llm, LLMWrapper  # LLMWrapper = ChatOllama alias
from .embedding import get_embedding, create_embedding, EmbeddingWrapper, OllamaEmbedding  # EmbeddingWrapper = OllamaEmbeddings alias

__all__ = [
    'get_config',
    'Config',
    'ConfigLoader',
    'get_llm',
    'create_llm',
    'LLMWrapper',
    'get_embedding',
    'create_embedding',
    'EmbeddingWrapper',
    'OllamaEmbedding',
]
