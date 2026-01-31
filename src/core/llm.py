"""LLM wrapper using LangChain for Modular RAG."""
import time
from typing import Any, Dict, List, Optional, Union
from langchain_ollama import ChatOllama, OllamaLLM as _OllamaLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
import logging

from .config import get_llm_config, LLMConfig

logger = logging.getLogger(__name__)

# Export OllamaLLM for backward compatibility
OllamaLLM = _OllamaLLM


class LLMWrapper:
    """LangChain-based LLM wrapper for Ollama."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM wrapper with LangChain."""
        self.config = config or get_llm_config()
        
        # Create LangChain ChatOllama LLM
        self.llm = ChatOllama(
            model=self.config.model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
        )
        
        logger.info(f"Initialized LangChain ChatOllama LLM: {self.config.model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt using LangChain."""
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        # Update LLM parameters
        self.llm.temperature = temperature
        self.llm.num_predict = max_tokens
        
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time() - start_time
            logger.info(f"LLM generation completed in {elapsed:.2f}s")
            # ChatOllama returns AIMessage, extract content
            if hasattr(response, 'content'):
                return response.content
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise
    
    def generate_with_history(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate text from a conversation history using LangChain."""
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        self.llm.temperature = temperature
        self.llm.num_predict = max_tokens
        
        # Convert to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        start_time = time.time()
        try:
            response = self.llm.invoke(lc_messages)
            elapsed = time.time() - start_time
            logger.info(f"LLM chat generation completed in {elapsed:.2f}s")
            # ChatOllama returns AIMessage, extract content
            if hasattr(response, 'content'):
                return response.content
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"LLM chat generation error: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # LangChain's Ollama doesn't expose token count directly
        # Using rough estimate: ~4 characters per token
        return len(text) // 4
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Ollama's embeddings API."""
        # Note: Ollama's embeddings API is separate from the LLM API
        # This would need to be implemented separately
        raise NotImplementedError("Use EmbeddingWrapper for embeddings")
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured output using LangChain's structured output."""
        from langchain.output_parsers import StructuredOutputParser, ResponseSchema
        
        # Convert schema to response schemas
        response_schemas = []
        for key, value_type in schema.items():
            if isinstance(value_type, str):
                type_str = value_type
            else:
                type_str = "string"
            
            response_schemas.append(ResponseSchema(name=key, description=key, type=type_str))
        
        # Create parser
        parser = StructuredOutputParser(response_schemas=response_schemas)
        
        # Get format instructions
        format_instructions = parser.get_format_instructions()
        
        # Add to prompt
        full_prompt = f"{prompt}\n\n{format_instructions}"
        
        # Generate
        response = self.generate(full_prompt, **kwargs)
        
        # Parse
        try:
            return parser.parse(response)
        except Exception:
            return {"raw_response": response}
    
    # Additional LangChain integration methods
    
    def create_prompt_template(self, template: str, input_variables: List[str]):
        """Create a LangChain PromptTemplate."""
        from langchain_core.prompts import PromptTemplate
        return PromptTemplate(template=template, input_variables=input_variables)
    
    def create_chain(self, prompt, output_parser=None):
        """Create a LangChain LLMChain."""
        from langchain.chains import LLMChain
        return LLMChain(llm=self.llm, prompt=prompt, output_parser=output_parser)


class LLMWrapperWithCallbacks(LLMWrapper):
    """LLM wrapper with callback support for monitoring."""
    
    def __init__(self, config: Optional[LLMConfig] = None, callbacks: Optional[List[BaseCallbackHandler]] = None):
        """Initialize with custom callbacks."""
        super().__init__(config)
        
        if callbacks:
            self.llm.callback_manager = CallbackManager(callbacks)
    
    def generate_with_callbacks(self, prompt: str, **kwargs) -> str:
        """Generate with callback tracking."""
        self.llm.callback_manager = CallbackManager(kwargs.get("callbacks", []))
        return self.generate(prompt, **kwargs)


# Singleton instance
_llm_instance: Optional[LLMWrapper] = None


def get_llm() -> LLMWrapper:
    """Get the global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance
