"""Answer generation using LangChain chains for Modular RAG.

Replaces custom AnswerGenerator and ResponseSynthesizer with:
- langchain.chains.combine_documents.create_stuff_documents_chain
- LangChain LCEL (prompt | llm | parser) for flexible composition
"""
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LangChainDocument
from langchain_core.language_models import BaseChatModel
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)


class AnswerResult(BaseModel):
    """Represents the result of answer generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Prompt Templates
# ============================================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Use ONLY the provided context to answer the question.
If the context does not contain enough information, say so clearly.
Be precise and cite specific details from the context."""),
    ("human", """Context:
{context}

Question: {input}

Answer:"""),
])

SYNTHESIS_MERGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are synthesizing multiple answers into one coherent response.
Combine the key information from all answers, resolve any contradictions, and provide
a comprehensive yet concise final answer."""),
    ("human", """Here are the individual answers to combine:

{context}

Original Question: {input}

Synthesized Answer:"""),
])

CONFIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a confidence scorer. Rate how well the answer addresses the question on a scale 0.0 to 1.0."),
    ("human", """Question: {question}
Answer: {answer}
Context Available: {has_context}

Score (just the number, 0.0 to 1.0):"""),
])


# ============================================================================
# Main Answer Generator using LangChain chains
# ============================================================================

class AnswerGenerator:
    """Answer generator using LangChain's create_stuff_documents_chain.
    
    Uses LangChain built-in document stuffing chain instead of custom
    prompt building and generation logic.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        llm_wrapper=None,  # Backward compatibility parameter
    ):
        """Initialize the answer generator.
        
        Args:
            llm: LangChain chat model (e.g., ChatOllama)
            prompt: Optional custom prompt template
            llm_wrapper: Deprecated, kept for backward compatibility
        """
        if llm is None:
            if llm_wrapper is not None:
                # Backward compat: llm_wrapper is now just ChatOllama
                self.llm = llm_wrapper
            else:
                from src.core.llm import get_llm
                self.llm = get_llm()
        else:
            self.llm = llm
        
        self.prompt = prompt or RAG_PROMPT
        
        # Create the LangChain stuff documents chain
        self._chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt,
        )
        
        logger.info("Initialized AnswerGenerator with LangChain create_stuff_documents_chain")
    
    def generate_answer(
        self,
        query: str,
        context_docs: List[LangChainDocument],
    ) -> AnswerResult:
        """Generate an answer from retrieved documents using LangChain chain.
        
        Args:
            query: The user's question
            context_docs: List of LangChain Document objects as context
            
        Returns:
            AnswerResult with the generated answer
        """
        if not context_docs:
            return AnswerResult(
                answer="I don't have enough context to answer this question.",
                confidence=0.0,
            )
        
        try:
            # Use the LangChain stuff documents chain
            answer = self._chain.invoke({
                "input": query,
                "context": context_docs,
            })
            
            # Build sources from documents
            sources = [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata,
                    "id": doc.metadata.get("id", ""),
                    "score": doc.metadata.get("score", 0.0),
                }
                for doc in context_docs
            ]
            
            # Estimate confidence
            confidence = self._estimate_confidence(query, answer, bool(context_docs))
            
            return AnswerResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return AnswerResult(
                answer=f"Error generating answer: {str(e)}",
                confidence=0.0,
            )
    
    def generate_answer_from_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> AnswerResult:
        """Generate answer from search result dicts (backward compat).
        
        Args:
            query: The user's question
            results: List of search result dicts with 'content' key
        """
        # Convert dicts to LangChain Documents
        context_docs = [
            LangChainDocument(
                page_content=r.get("content", ""),
                metadata={
                    "id": r.get("id", ""),
                    "score": r.get("score", 0.0),
                    **(r.get("metadata", {})),
                },
            )
            for r in results
        ]
        return self.generate_answer(query, context_docs)
    
    def _estimate_confidence(self, query: str, answer: str, has_context: bool) -> float:
        """Estimate confidence score for the answer."""
        try:
            chain = CONFIDENCE_PROMPT | self.llm | StrOutputParser()
            result = chain.invoke({
                "question": query,
                "answer": answer[:500],
                "has_context": str(has_context),
            })
            
            import re
            score_match = re.search(r'(\d+\.?\d*)', result)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"Confidence estimation failed: {e}")
        
        return 0.5 if has_context else 0.2


# ============================================================================
# Response Synthesizer using LangChain chains
# ============================================================================

class ResponseSynthesizer:
    """Synthesize multiple answers using LangChain chains.
    
    Replaces custom synthesis logic with LangChain document chains.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        llm_wrapper=None,  # Backward compatibility
    ):
        if llm is None:
            if llm_wrapper is not None:
                self.llm = llm_wrapper
            else:
                from src.core.llm import get_llm
                self.llm = get_llm()
        else:
            self.llm = llm
    
    def synthesize(
        self,
        query: str,
        answers: List[str],
        strategy: str = "merge",
    ) -> str:
        """Synthesize multiple answers into one.
        
        Args:
            query: Original question
            answers: List of individual answers
            strategy: Synthesis strategy ('merge', 'best', 'concatenate')
            
        Returns:
            Synthesized answer
        """
        if not answers:
            return "No answers to synthesize."
        
        if len(answers) == 1:
            return answers[0]
        
        if strategy == "concatenate":
            return "\n\n".join(answers)
        
        elif strategy == "best":
            return answers[0]
        
        elif strategy == "merge":
            # Use LangChain stuff chain to merge answers
            answer_docs = [
                LangChainDocument(
                    page_content=f"Answer {i+1}: {answer}",
                    metadata={"answer_index": i},
                )
                for i, answer in enumerate(answers)
            ]

            merge_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=SYNTHESIS_MERGE_PROMPT,
            )

            return merge_chain.invoke({
                "input": query,
                "context": answer_docs,
            })

        else:
            return "\n\n".join(answers)
