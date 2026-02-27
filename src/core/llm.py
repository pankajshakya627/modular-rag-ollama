"""LLM factory using LangChain ChatOllama for Modular RAG.

Instead of a custom wrapper, this module provides a factory function
that returns a properly configured ChatOllama instance from langchain-ollama.
All LLM functionality (invoke, stream, batch, structured output, etc.)
is provided directly by the LangChain ChatOllama class.
"""
from typing import Optional
from langchain_ollama import ChatOllama
import logging

from .config import get_llm_config, LLMConfig

logger = logging.getLogger(__name__)


def create_llm(config: Optional[LLMConfig] = None) -> ChatOllama:
    """Create a configured ChatOllama instance.
    
    Returns a LangChain ChatOllama that supports:
    - .invoke(prompt_or_messages) -> AIMessage
    - .stream(prompt_or_messages) -> Iterator[AIMessageChunk]
    - .batch(inputs) -> List[AIMessage]
    - .with_structured_output(schema) -> Runnable
    - .get_num_tokens(text) -> int
    - LCEL composition: prompt | llm | parser
    """
    config = config or get_llm_config()
    
    llm = ChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )
    
    logger.info(f"Initialized ChatOllama LLM: {config.model}")
    return llm


# Singleton instance
_llm_instance: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    """Get the global ChatOllama LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance


# Backward compatibility aliases
LLMWrapper = ChatOllama  # Type alias for imports that reference LLMWrapper
