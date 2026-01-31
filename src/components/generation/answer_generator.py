"""Answer generation and response synthesis components."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..retrieval.vector_store import SearchResult
from ..reranking.base import RerankedResult
from ...core.llm import LLMWrapper

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Represents the result of answer generation."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerGenerator:
    """Generates answers using retrieved context."""
    
    def __init__(
        self,
        llm_wrapper: Optional[LLMWrapper] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the answer generator."""
        self.llm_wrapper = llm_wrapper or LLMWrapper()
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources when making claims. If the context doesn't contain enough 
information to answer the question, say so clearly. Be concise and accurate."""
    
    def generate_answer(
        self,
        query: str,
        context: str,
        sources: Optional[List[SearchResult]] = None,
        max_length: Optional[int] = None,
    ) -> AnswerResult:
        """Generate an answer given a query and context."""
        # Build the prompt
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        try:
            if self.system_prompt:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
                answer = self.llm_wrapper.llm.generate_with_history(messages)
            else:
                answer = self.llm_wrapper.llm.generate(prompt)
            
            # Prepare sources
            source_list = []
            if sources:
                for src in sources:
                    source_list.append({
                        "id": src.id,
                        "content": src.content[:200] + "..." if len(src.content) > 200 else src.content,
                        "score": src.score,
                        "metadata": src.metadata,
                    })
            
            # Estimate confidence based on answer length and structure
            confidence = self._estimate_confidence(answer, context)
            
            return AnswerResult(
                answer=answer.strip(),
                sources=source_list,
                confidence=confidence,
                metadata={
                    "query": query,
                    "context_length": len(context),
                    "num_sources": len(sources) if sources else 0,
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return AnswerResult(
                answer="I apologize, but I encountered an error while generating the answer.",
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for answer generation."""
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate confidence based on answer characteristics."""
        # Basic heuristics
        confidence = 0.5
        
        # Longer answers tend to be more confident
        if len(answer) > 100:
            confidence += 0.2
        if len(answer) > 300:
            confidence += 0.1
        
        # Check for uncertainty markers
        uncertainty_markers = ["I don't know", "not sure", "uncertain", "might be", "possibly"]
        for marker in uncertainty_markers:
            if marker.lower() in answer.lower():
                confidence -= 0.1
        
        # Check for direct answers
        if "based on the context" in answer.lower():
            confidence += 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def generate_answer_from_results(
        self,
        query: str,
        results: List[SearchResult],
        max_context_length: int = 4000,
    ) -> AnswerResult:
        """Generate an answer from search results."""
        # Build context from results
        context_parts = []
        total_length = 0
        
        for result in results:
            if total_length + len(result.content) > max_context_length:
                break
            context_parts.append(result.content)
            total_length += len(result.content)
        
        context = "\n\n".join(context_parts)
        
        return self.generate_answer(query, context, sources=results)
    
    def generate_answer_from_reranked(
        self,
        query: str,
        results: List[RerankedResult],
        max_context_length: int = 4000,
    ) -> AnswerResult:
        """Generate an answer from reranked results."""
        # Convert to SearchResult-like format
        search_results = [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.reranked_score,
                metadata=r.metadata,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
            )
            for r in results
        ]
        
        return self.generate_answer_from_results(query, search_results, max_context_length)


class ResponseSynthesizer:
    """Synthesizes comprehensive responses from multiple retrieval passes."""
    
    def __init__(self, llm_wrapper: Optional[LLMWrapper] = None):
        """Initialize the response synthesizer."""
        self.llm_wrapper = llm_wrapper or LLMWrapper()
    
    def synthesize(
        self,
        query: str,
        retrieval_results: Dict[str, List[SearchResult]],
        synthesis_strategy: str = "concatenate",
    ) -> AnswerResult:
        """Synthesize a response from multiple retrieval methods."""
        if synthesis_strategy == "concatenate":
            return self._synthesize_concat(query, retrieval_results)
        elif synthesis_strategy == "merge":
            return self._synthesize_merge(query, retrieval_results)
        elif synthesis_strategy == "selective":
            return self._synthesize_selective(query, retrieval_results)
        else:
            return self._synthesize_concat(query, retrieval_results)
    
    def _synthesize_concat(
        self,
        query: str,
        retrieval_results: Dict[str, List[SearchResult]],
    ) -> AnswerResult:
        """Concatenate results from all methods and synthesize."""
        all_content = []
        all_sources = []
        
        for method, results in retrieval_results.items():
            for result in results:
                all_content.append(f"[{method}] {result.content}")
                all_sources.append({
                    **result.to_dict(),
                    "retrieval_method": method,
                })
        
        context = "\n\n".join(all_content[:10])  # Limit to top 10
        
        prompt = f"""Synthesize a comprehensive answer from multiple retrieval sources.
Different retrieval methods may have found different relevant passages.

Sources:
{context}

Question: {query}

Please provide a synthesized answer that combines information from all relevant sources:"""
        
        try:
            answer = self.llm_wrapper.llm.generate(prompt, max_tokens=1024)
            
            return AnswerResult(
                answer=answer.strip(),
                sources=all_sources[:10],
                confidence=0.7,
                metadata={"synthesis_strategy": "concatenate", "methods_used": list(retrieval_results.keys())}
            )
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return AnswerResult(answer="Error synthesizing response", confidence=0.0)
    
    def _synthesize_merge(
        self,
        query: str,
        retrieval_results: Dict[str, List[SearchResult]],
    ) -> AnswerResult:
        """Merge answers from each method then synthesize."""
        # Get answers from each method
        answers = {}
        for method, results in retrieval_results.items():
            if results:
                generator = AnswerGenerator(self.llm_wrapper)
                answers[method] = generator.generate_answer_from_results(query, results[:3])
        
        # Synthesize final answer
        combined_answers = "\n\n".join([
            f"Method: {method}\nAnswer: {ans.answer}"
            for method, ans in answers.items()
        ])
        
        prompt = f"""Given the following answers from different retrieval methods,
synthesize a single comprehensive answer.

{combined_answers}

Original Question: {query}

Synthesized Answer:"""
        
        try:
            answer = self.llm_wrapper.llm.generate(prompt, max_tokens=1024)
            
            all_sources = []
            for method, ans in answers.items():
                all_sources.extend(ans.sources)
            
            return AnswerResult(
                answer=answer.strip(),
                sources=all_sources[:10],
                confidence=0.75,
                metadata={"synthesis_strategy": "merge", "methods_used": list(retrieval_results.keys())}
            )
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return AnswerResult(answer="Error synthesizing response", confidence=0.0)
    
    def _synthesize_selective(
        self,
        query: str,
        retrieval_results: Dict[str, List[SearchResult]],
    ) -> AnswerResult:
        """Select best results from each method and synthesize."""
        # Take top result from each method
        selected_results = {}
        for method, results in retrieval_results.items():
            if results:
                # Sort by score and take top
                sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
                selected_results[method] = sorted_results[:2]
        
        # Flatten and sort by score
        all_results = []
        for method, results in selected_results.items():
            for result in results:
                all_results.append({
                    **result.to_dict(),
                    "retrieval_method": method,
                })
        
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Build context from top results
        context_parts = [
            f"[{r['retrieval_method']}] {r['content']}"
            for r in all_results[:5]
        ]
        context = "\n\n".join(context_parts)
        
        prompt = f"""Synthesize an answer from the most relevant passages.

{context}

Question: {query}

Synthesized Answer:"""
        
        try:
            answer = self.llm_wrapper.llm.generate(prompt, max_tokens=1024)
            
            return AnswerResult(
                answer=answer.strip(),
                sources=all_results[:5],
                confidence=0.8,
                metadata={"synthesis_strategy": "selective", "methods_used": list(selected_results.keys())}
            )
        except Exception as e:
            logger.error(f"Error synthesizing response: {e}")
            return AnswerResult(answer="Error synthesizing response", confidence=0.0)
