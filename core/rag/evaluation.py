# core/rag/evaluation.py

"""RAG evaluation framework for measuring pipeline performance."""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
import uuid
from datetime import datetime

from langchain.schema import Document
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("rag_evaluation")

@dataclass
class EvaluationResult:
    """Results from a RAG evaluation."""
    query_id: str
    query: str
    response: str
    retrieval_time: float
    generation_time: float
    total_time: float
    document_count: int
    document_ids: List[str] = field(default_factory=list)
    document_scores: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class RAGEvaluator:
    """Evaluator for measuring RAG pipeline performance."""
    
    def __init__(self, 
                 save_dir: Optional[str] = None,
                 metrics: Optional[List[str]] = None):
        """Initialize the evaluator.
        
        Args:
            save_dir: Directory to save evaluation results
            metrics: List of metrics to compute
        """
        self.save_dir = save_dir or "rag_evaluation"
        self.metrics = metrics or ["retrieval_precision", "answer_relevance"]
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize evaluation results
        self.results = []
        
        logger.debug(f"Initialized RAG evaluator with metrics: {self.metrics}")
    
    def evaluate_query(self, 
                       query: str,
                       rag_pipeline: Any,
                       ground_truth: Optional[str] = None,
                       expected_doc_ids: Optional[List[str]] = None) -> EvaluationResult:
        """Evaluate a single query.
        
        Args:
            query: The query to evaluate
            rag_pipeline: The RAG pipeline to evaluate
            ground_truth: Optional ground truth answer
            expected_doc_ids: Optional list of expected document IDs
            
        Returns:
            Evaluation result
        """
        # Generate a query ID
        query_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Track retrieval time
        retrieval_start = time.time()
        context, source_docs = rag_pipeline._retrieval_step(query)
        retrieval_time = time.time() - retrieval_start
        
        # Track generation time
        generation_start = time.time()
        result = rag_pipeline.run(query)
        generation_time = time.time() - generation_start
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Extract retrieved document IDs and scores
        doc_ids = []
        doc_scores = []
        for doc in source_docs:
            doc_id = doc.metadata.get('id', None)
            if doc_id:
                doc_ids.append(doc_id)
            # Some implementations store relevance scores in metadata
            score = doc.metadata.get('score', None)
            if score is not None:
                doc_scores.append(score)
        
        # Compute metrics
        metrics = self._compute_metrics(
            query=query,
            response=result['result'],
            retrieved_docs=source_docs,
            ground_truth=ground_truth,
            expected_doc_ids=expected_doc_ids
        )
        
        # Create evaluation result
        eval_result = EvaluationResult(
            query_id=query_id,
            query=query,
            response=result['result'],
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            document_count=len(source_docs),
            document_ids=doc_ids,
            document_scores=doc_scores,
            metrics=metrics
        )
        
        # Add to results
        self.results.append(eval_result)
        
        # Save result
        self._save_result(eval_result)
        
        return eval_result
    
    def evaluate_dataset(self, 
                         queries: List[str],
                         rag_pipeline: Any,
                         ground_truths: Optional[List[str]] = None,
                         expected_doc_ids: Optional[List[List[str]]] = None) -> List[EvaluationResult]:
        """Evaluate a dataset of queries.
        
        Args:
            queries: List of queries to evaluate
            rag_pipeline: The RAG pipeline to evaluate
            ground_truths: Optional list of ground truth answers
            expected_doc_ids: Optional list of lists of expected document IDs
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Process each query
        for i, query in enumerate(queries):
            # Get corresponding ground truth and expected docs if available
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            expected_docs = expected_doc_ids[i] if expected_doc_ids and i < len(expected_doc_ids) else None
            
            # Evaluate query
            result = self.evaluate_query(
                query=query,
                rag_pipeline=rag_pipeline,
                ground_truth=ground_truth,
                expected_doc_ids=expected_docs
            )
            
            results.append(result)
            
            # Log progress
            logger.info(f"Evaluated query {i+1}/{len(queries)}: {query[:50]}...")
        
        # Compute and log aggregate metrics
        self._log_aggregate_metrics(results)
        
        return results
    
    def _compute_metrics(self,
                         query: str,
                         response: str,
                         retrieved_docs: List[Document],
                         ground_truth: Optional[str] = None,
                         expected_doc_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            query: The query
            response: The response
            retrieved_docs: Retrieved documents
            ground_truth: Optional ground truth answer
            expected_doc_ids: Optional list of expected document IDs
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Simple retrieval precision if expected_doc_ids is provided
        if "retrieval_precision" in self.metrics and expected_doc_ids:
            # Get retrieved document IDs
            retrieved_ids = []
            for doc in retrieved_docs:
                doc_id = doc.metadata.get('id', None)
                if doc_id:
                    retrieved_ids.append(doc_id)
            
            # Calculate precision
            if expected_doc_ids and retrieved_ids:
                correct = sum(1 for doc_id in retrieved_ids if doc_id in expected_doc_ids)
                metrics["retrieval_precision"] = correct / len(retrieved_ids) if retrieved_ids else 0
            else:
                metrics["retrieval_precision"] = 0
        
        # For a real implementation, you would integrate with an LLM to compute answer relevance
        # Here we're just providing a placeholder
        if "answer_relevance" in self.metrics:
            # Placeholder for answer relevance score (0-1)
            # In a real implementation, this would use an LLM to evaluate answer quality
            metrics["answer_relevance"] = 0.5
        
        return metrics
    
    def _save_result(self, result: EvaluationResult):
        """Save evaluation result to file.
        
        Args:
            result: Evaluation result
        """
        # Create filename
        filename = f"{result.query_id}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _log_aggregate_metrics(self, results: List[EvaluationResult]):
        """Compute and log aggregate metrics.
        
        Args:
            results: List of evaluation results
        """
        if not results:
            logger.warning("No results to compute aggregate metrics")
            return
        
        # Compute averages
        avg_retrieval_time = sum(r.retrieval_time for r in results) / len(results)
        avg_generation_time = sum(r.generation_time for r in results) / len(results)
        avg_total_time = sum(r.total_time for r in results) / len(results)
        avg_doc_count = sum(r.document_count for r in results) / len(results)
        
        # Compute average metrics
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
        
        avg_metrics = {}
        for key in metric_keys:
            values = [r.metrics.get(key, 0) for r in results if key in r.metrics]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Log results
        logger.info(f"Evaluation complete: {len(results)} queries")
        logger.info(f"Average retrieval time: {avg_retrieval_time:.4f}s")
        logger.info(f"Average generation time: {avg_generation_time:.4f}s")
        logger.info(f"Average total time: {avg_total_time:.4f}s")
        logger.info(f"Average document count: {avg_doc_count:.2f}")
        
        for key, value in avg_metrics.items():
            logger.info(f"Average {key}: {value:.4f}")
        
        # Save aggregate metrics
        self._save_aggregate_metrics(results)
    
    def _save_aggregate_metrics(self, results: List[EvaluationResult]):
        """Save aggregate metrics to file.
        
        Args:
            results: List of evaluation results
        """
        # Compute aggregates
        metrics = {
            "query_count": len(results),
            "avg_retrieval_time": sum(r.retrieval_time for r in results) / len(results),
            "avg_generation_time": sum(r.generation_time for r in results) / len(results),
            "avg_total_time": sum(r.total_time for r in results) / len(results),
            "avg_document_count": sum(r.document_count for r in results) / len(results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add average specific metrics
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
        
        for key in metric_keys:
            values = [r.metrics.get(key, 0) for r in results if key in r.metrics]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)
        
        # Save as JSON
        filepath = os.path.join(self.save_dir, "aggregate_metrics.json")
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

def create_evaluation_queries() -> List[Dict[str, Any]]:
    """Create a set of test queries for RAG evaluation.
    
    Returns:
        List of query dictionaries with query, ground_truth, and expected_docs
    """
    return [
        {
            "query": "What are the benefits of using ChromaDB versus FAISS?",
            "ground_truth": "ChromaDB offers persistence, metadata filtering, and a simple API, while FAISS provides high performance and scalability for large datasets.",
            "expected_docs": []  # Would be populated with known relevant document IDs
        },
        {
            "query": "How does the RAG pipeline handle errors during document retrieval?",
            "ground_truth": "The RAG pipeline includes error handling in the retrieval step, falling back to empty context when retrieval fails.",
            "expected_docs": []
        },
        {
            "query": "What components are needed for a minimal RAG implementation?",
            "ground_truth": "A minimal RAG implementation requires a document store, embedding model, retriever, and language model for generation.",
            "expected_docs": []
        }
    ]