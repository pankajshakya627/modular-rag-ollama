"""Evaluation metrics for Modular RAG."""
import re
from typing import Any, Dict, List, Optional
import logging

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

from ..components.retrieval.vector_store import SearchResult
from ..components.generation.answer_generator import AnswerResult

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Represents evaluation results."""
    metric_name: str
    score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


class RetrievalEvalResult(BaseModel):
    """Represents retrieval evaluation results."""
    query: str
    relevant_docs: List[str] = Field(default_factory=list)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)
    mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    ndcg: float = Field(default=0.0, ge=0.0, le=1.0)


class GenerationEvalResult(BaseModel):
    """Represents generation evaluation results."""
    query: str
    answer: str
    context: str
    answer_relevancy: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    rouge_scores: Dict[str, float] = Field(default_factory=dict)


def compute_precision(
    retrieved_docs: List[SearchResult],
    relevant_doc_ids: List[str],
    k: Optional[int] = None,
) -> float:
    """Compute precision at k."""
    if k is None:
        k = len(retrieved_docs)
    
    retrieved_ids = [doc.id for doc in retrieved_docs[:k]]
    relevant_set = set(relevant_doc_ids)
    
    if not relevant_set:
        return 0.0
    
    true_positives = len(set(retrieved_ids) & relevant_set)
    retrieved_count = len(retrieved_ids)
    
    if retrieved_count == 0:
        return 0.0
    
    return true_positives / retrieved_count


def compute_recall(
    retrieved_docs: List[SearchResult],
    relevant_doc_ids: List[str],
    k: Optional[int] = None,
) -> float:
    """Compute recall at k."""
    if k is None:
        k = len(retrieved_docs)
    
    retrieved_ids = set(doc.id for doc in retrieved_docs[:k])
    relevant_set = set(relevant_doc_ids)
    
    if not relevant_set:
        return 0.0
    
    true_positives = len(retrieved_ids & relevant_set)
    relevant_count = len(relevant_set)
    
    if relevant_count == 0:
        return 0.0
    
    return true_positives / relevant_count


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_mrr(retrieved_docs: List[SearchResult], relevant_doc_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    relevant_set = set(relevant_doc_ids)
    
    for i, doc in enumerate(retrieved_docs):
        if doc.id in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0


def compute_ndcg(retrieved_docs: List[SearchResult], relevant_doc_ids: List[str], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    relevant_set = set(relevant_doc_ids)
    
    # Calculate DCG
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc.id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate ideal DCG
    ideal_relevant = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_answer_relevancy(
    query: str,
    answer: str,
    llm_wrapper,
) -> float:
    """Compute answer relevancy using LLM-based scoring via invoke()."""
    from langchain_core.messages import HumanMessage

    prompt = (
        f"Rate how relevant the following answer is to the query on a scale of 0 to 1.\n\n"
        f"Query: {query}\n\nAnswer: {answer}\n\nRelevance Score (0-1):"
    )
    try:
        response = llm_wrapper.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        score_match = re.search(r"(\d+\.?\d*)", text)
        if score_match:
            return max(0.0, min(1.0, float(score_match.group(1))))
        return 0.5
    except Exception as e:
        logger.error(f"Error computing answer relevancy: {e}")
        return 0.5


def compute_faithfulness(
    query: str,
    answer: str,
    context: str,
    llm_wrapper,
) -> float:
    """Compute faithfulness via LLM invoke()."""
    from langchain_core.messages import HumanMessage

    prompt = (
        f"Rate how faithful the following answer is to the context on a scale of 0 to 1.\n"
        f"An answer is faithful if all claims are supported by the context.\n\n"
        f"Context: {context[:1000]}...\n\nAnswer: {answer}\n\nFaithfulness Score (0-1):"
    )
    try:
        response = llm_wrapper.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        score_match = re.search(r"(\d+\.?\d*)", text)
        if score_match:
            return max(0.0, min(1.0, float(score_match.group(1))))
        return 0.5
    except Exception as e:
        logger.error(f"Error computing faithfulness: {e}")
        return 0.5


def compute_context_precision(
    query: str,
    retrieved_docs: List[SearchResult],
    relevant_doc_ids: List[str],
    llm_wrapper,
) -> float:
    """Compute context precision using LLM invoke()."""
    from langchain_core.messages import HumanMessage

    context = "\n\n".join([
        f"[Doc {i+1}] {doc.content[:500]}"
        for i, doc in enumerate(retrieved_docs[:5])
    ])
    prompt = (
        f"Evaluate the precision of the retrieved context for the query.\n"
        f"Rate on a scale of 0 to 1 (1 = all retrieved docs relevant).\n\n"
        f"Query: {query}\n\nRetrieved Context:\n{context}\n\nPrecision Score (0-1):"
    )
    try:
        response = llm_wrapper.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        score_match = re.search(r"(\d+\.?\d*)", text)
        if score_match:
            return max(0.0, min(1.0, float(score_match.group(1))))
        return 0.5
    except Exception as e:
        logger.error(f"Error computing context precision: {e}")
        return 0.5


class RetrievalEvaluator:
    """Evaluator for retrieval components."""
    
    def __init__(self, llm_wrapper=None):
        """Initialize the evaluator."""
        self.llm_wrapper = llm_wrapper
    
    def evaluate(
        self,
        queries: List[str],
        retrieved_docs_per_query: List[List[SearchResult]],
        relevant_docs_per_query: List[List[str]],
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        results = {
            "precision": {},
            "recall": {},
            "f1": {},
            "mrr": {},
            "ndcg": {},
        }
        
        for k in k_values:
            results["precision"][f"@{k}"] = []
            results["recall"][f"@{k}"] = []
            results["f1"][f"@{k}"] = []
        
        results["mrr"] = []
        results["ndcg"][f"@{max(k_values)}"] = []
        
        for query, retrieved, relevant in zip(
            queries, retrieved_docs_per_query, relevant_docs_per_query
        ):
            # Compute metrics at each k
            for k in k_values:
                precision = compute_precision(retrieved, relevant, k)
                recall = compute_recall(retrieved, relevant, k)
                f1 = compute_f1(precision, recall)
                
                results["precision"][f"@{k}"].append(precision)
                results["recall"][f"@{k}"].append(recall)
                results["f1"][f"@{k}"].append(f1)
            
            # Compute MRR
            mrr = compute_mrr(retrieved, relevant)
            results["mrr"].append(mrr)
            
            # Compute NDCG at max k
            ndcg = compute_ndcg(retrieved, relevant, max(k_values))
            results["ndcg"][f"@{max(k_values)}"].append(ndcg)
        
        # Average results
        averaged_results = {}
        for metric, values in results.items():
            if isinstance(values, dict):
                averaged_results[metric] = {
                    k: np.mean(v) for k, v in values.items()
                }
            else:
                averaged_results[metric] = np.mean(values)
        
        return averaged_results


class GenerationEvaluator:
    """Evaluator for generation components."""
    
    def __init__(self, llm_wrapper):
        """Initialize the evaluator."""
        self.llm_wrapper = llm_wrapper
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def evaluate(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[str],
    ) -> Dict[str, float]:
        """Evaluate generation performance."""
        results = {
            "answer_relevancy": [],
            "faithfulness": [],
            "context_precision": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
        }
        
        for query, answer, context in zip(queries, answers, contexts):
            # Compute metrics
            relevancy = compute_answer_relevancy(query, answer, self.llm_wrapper)
            faithfulness = compute_faithfulness(query, answer, context, self.llm_wrapper)
            ctx_precision = compute_context_precision(
                query,
                [SearchResult(id="ctx", content=context, score=1.0)],
                ["ctx"],
                self.llm_wrapper,
            )
            
            results["answer_relevancy"].append(relevancy)
            results["faithfulness"].append(faithfulness)
            results["context_precision"].append(ctx_precision)
            
            # Compute ROUGE scores (if reference available)
            # This would need a reference answer
            results["rouge1"].append(0.0)
            results["rouge2"].append(0.0)
            results["rougeL"].append(0.0)
        
        # Average results
        return {metric: np.mean(values) for metric, values in results.items()}


class RAGEvaluator:
    """Comprehensive RAG evaluator."""
    
    def __init__(self, llm_wrapper):
        """Initialize the evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator(llm_wrapper)
        self.generation_evaluator = GenerationEvaluator(llm_wrapper)
    
    def evaluate_sample(
        self,
        query: str,
        answer_result: AnswerResult,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single RAG sample."""
        # Build context from sources
        context = "\n\n".join([
            src.get("content", "") for src in answer_result.sources[:5]
        ])
        
        # Compute generation metrics
        gen_metrics = self.generation_evaluator.evaluate(
            queries=[query],
            answers=[answer_result.answer],
            contexts=[context],
        )
        
        # Compute retrieval metrics if relevant docs provided
        retrieval_metrics = {}
        if relevant_doc_ids:
            retrieved_docs = [
                SearchResult(
                    id=src.get("id", ""),
                    content=src.get("content", ""),
                    score=src.get("score", 0.0),
                    metadata=src.get("metadata", {}),
                )
                for src in answer_result.sources
            ]
            
            retrieval_metrics = self.retrieval_evaluator.evaluate(
                queries=[query],
                retrieved_docs_per_query=[retrieved_docs],
                relevant_docs_per_query=[relevant_doc_ids],
            )
        
        return {
            "query": query,
            "answer": answer_result.answer,
            "confidence": answer_result.confidence,
            "generation_metrics": gen_metrics,
            "retrieval_metrics": retrieval_metrics,
        }
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate a dataset of RAG samples."""
        if sample_size:
            dataset = dataset[:sample_size]
        
        results = []
        for sample in dataset:
            result = self.evaluate_sample(
                query=sample["query"],
                answer_result=AnswerResult(
                    answer=sample.get("answer", ""),
                    sources=sample.get("sources", []),
                    confidence=sample.get("confidence", 0.0),
                ),
                relevant_doc_ids=sample.get("relevant_docs"),
            )
            results.append(result)
        
        # Aggregate results
        aggregated = {
            "num_samples": len(results),
            "avg_confidence": np.mean([r["confidence"] for r in results]),
            "generation_metrics": {},
            "retrieval_metrics": {},
        }
        
        # Aggregate generation metrics
        gen_metrics_list = [r["generation_metrics"] for r in results]
        for metric in ["answer_relevancy", "faithfulness", "context_precision"]:
            aggregated["generation_metrics"][metric] = np.mean([
                m.get(metric, 0) for m in gen_metrics_list
            ])
        
        # Aggregate retrieval metrics
        ret_metrics_list = [r["retrieval_metrics"] for r in results if r["retrieval_metrics"]]
        if ret_metrics_list:
            for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                if metric in ret_metrics_list[0]:
                    if isinstance(ret_metrics_list[0][metric], dict):
                        aggregated["retrieval_metrics"][metric] = {
                            k: np.mean([m[metric][k] for m in ret_metrics_list])
                            for k in ret_metrics_list[0][metric].keys()
                        }
                    else:
                        aggregated["retrieval_metrics"][metric] = np.mean([
                            m.get(metric, 0) for m in ret_metrics_list
                        ])
        
        return aggregated
