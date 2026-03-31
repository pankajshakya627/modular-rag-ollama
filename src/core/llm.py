"""LLM factory using LangChain ChatOllama for Modular RAG.

Production improvements:
- Thread-safe singleton via functools.lru_cache (no mutable module globals)
- Retry logic via tenacity for transient Ollama connection failures
- Async-ready: same ChatOllama supports .ainvoke() / .astream()
"""
import functools
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .config import get_llm_config, LLMConfig

logger = logging.getLogger(__name__)


def _make_llm(config: LLMConfig) -> ChatOllama:
    """Internal factory — creates a configured ChatOllama instance.

    ChatOllama supports:
    - .invoke(messages)           → AIMessage
    - .ainvoke(messages)          → AIMessage   (async)
    - .stream(messages)           → Iterator[AIMessageChunk]
    - .astream(messages)          → AsyncIterator[AIMessageChunk]
    - .batch(inputs)              → List[AIMessage]
    - .with_structured_output(schema) → Runnable
    - LCEL: prompt | llm | parser
    """
    llm = ChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens,
    )
    logger.info(f"Initialized ChatOllama LLM: model={config.model}, base_url={config.base_url}")
    return llm


@functools.lru_cache(maxsize=4)
def _cached_llm(model: str, base_url: str, temperature: float, max_tokens: int) -> ChatOllama:
    """Thread-safe cached LLM factory keyed on config parameters.

    lru_cache guarantees only one instance per unique config combination,
    and is thread-safe by default in CPython.
    """
    from .config import LLMConfig
    cfg = LLMConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _make_llm(cfg)


def get_llm(config: Optional[LLMConfig] = None) -> ChatOllama:
    """Get a thread-safe cached ChatOllama LLM instance.

    Uses lru_cache keyed on config values — avoids mutable module-level singletons.
    Repeated calls with the same config return the exact same object.
    """
    cfg = config or get_llm_config()
    return _cached_llm(
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def create_llm(config: Optional[LLMConfig] = None) -> ChatOllama:
    """Create a new (non-cached) ChatOllama instance.

    Use this when you explicitly need a fresh instance (e.g. different temperature).
    """
    cfg = config or get_llm_config()
    return _make_llm(cfg)


@retry(
    retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def invoke_with_retry(llm: ChatOllama, messages) -> str:
    """Invoke the LLM with automatic retry on transient connection errors.

    Retries up to 3 times with exponential backoff (1s, 2s, up to 10s).
    Raises the last exception if all retries are exhausted.
    """
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


# Backward compatibility aliases
LLMWrapper = ChatOllama  # Type alias for imports that reference LLMWrapper
