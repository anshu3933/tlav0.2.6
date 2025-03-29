# core/rag/observability.py

from typing import Dict, Any, List, Optional, Callable, Union
import json
import time
import os
from functools import wraps
from datetime import datetime
from config.logging_config import get_module_logger

logger = get_module_logger("rag_observability")

class RagObservability:
    """Utilities for RAG pipeline observability."""
    
    def __init__(self, 
                 log_dir: Optional[str] = None, 
                 enable_timing: bool = True,
                 enable_logging: bool = True,
                 enable_tracing: bool = False):
        """Initialize observability utils.
        
        Args:
            log_dir: Directory for log files (default: logs/rag)
            enable_timing: Whether to record timing
            enable_logging: Whether to log events
            enable_tracing: Whether to enable tracing
        """
        self.log_dir = log_dir or "logs/rag"
        self.enable_timing = enable_timing
        self.enable_logging = enable_logging
        self.enable_tracing = enable_tracing
        
        # Create log directory if it doesn't exist
        if self.enable_logging and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize timing data
        self.timings = {}
        
        logger.debug(f"Initialized RAG observability with logging to {self.log_dir}")
    
    def rag_step_callback(self) -> Callable:
        """Create a callback function for RAG pipeline steps.
        
        Returns:
            Callback function
        """
        def callback(step: str, input: Any, output: Any):
            # Record timestamp
            timestamp = datetime.now().isoformat()
            
            # Log step
            if self.enable_logging:
                self._log_step(step, input, output, timestamp)
            
            # Record timing if it's the start or end of a run
            if self.enable_timing:
                if step == "start":
                    self.timings[self._get_query_id(input)] = {
                        "start_time": time.time(),
                        "steps": {}
                    }
                elif step == "end":
                    query_id = self._get_query_id(input)
                    if query_id in self.timings:
                        self.timings[query_id]["end_time"] = time.time()
                        self.timings[query_id]["total_time"] = (
                            self.timings[query_id]["end_time"] - 
                            self.timings[query_id]["start_time"]
                        )
                else:
                    # Record timing for intermediate steps
                    query_id = self._get_query_id(input)
                    if query_id in self.timings:
                        step_key = f"{step}_{int(time.time() * 1000)}"
                        self.timings[query_id]["steps"][step_key] = {
                            "timestamp": timestamp,
                            "step": step
                        }
        
        return callback
    
    def _log_step(self, step: str, input: Any, output: Any, timestamp: str):
        """Log a RAG pipeline step.
        
        Args:
            step: Step name
            input: Step input
            output: Step output
            timestamp: ISO timestamp
        """
        try:
            # Create log entry
            log_entry = {
                "timestamp": timestamp,
                "step": step,
                "input_type": type(input).__name__
            }
            
            # Add appropriate data based on step
            if step == "retrieval":
                log_entry["doc_count"] = len(output) if isinstance(output, list) else 0
            elif step == "generation":
                # Truncate output for logging
                output_str = str(output)
                log_entry["output_length"] = len(output_str)
                log_entry["output_sample"] = output_str[:100] + "..." if len(output_str) > 100 else output_str
            elif step == "end":
                if isinstance(output, dict) and "execution_time" in output:
                    log_entry["execution_time"] = output["execution_time"]
                    log_entry["doc_count"] = len(output.get("source_documents", []))
            
            # Write to log file
            query_id = self._get_query_id(input)
            log_file = os.path.join(self.log_dir, f"rag_{query_id}_{step}.json")
            
            with open(log_file, "w") as f:
                json.dump(log_entry, f, default=str)
                
        except Exception as e:
            logger.error(f"Error logging RAG step: {str(e)}")
    
    def _get_query_id(self, input: Any) -> str:
        """Generate a query ID from input.
        
        Args:
            input: Query input
            
        Returns:
            Query ID string
        """
        # Handle different input types
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict) and "question" in input:
            query = input["question"]
        else:
            query = str(input)
        
        # Create a simple hash
        query_hash = hash(query) % 10000
        timestamp = int(time.time())
        
        return f"{timestamp}_{query_hash}"
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of timing data.
        
        Returns:
            Dictionary with timing summary
        """
        if not self.enable_timing:
            return {"error": "Timing is not enabled"}
        
        # Calculate summary statistics
        summary = {
            "total_queries": len(self.timings),
            "average_time": 0,
            "max_time": 0,
            "min_time": float("inf") if self.timings else 0
        }
        
        if self.timings:
            total_times = []
            for query_id, timing in self.timings.items():
                if "total_time" in timing:
                    total_time = timing["total_time"]
                    total_times.append(total_time)
                    
                    # Update max and min
                    summary["max_time"] = max(summary["max_time"], total_time)
                    summary["min_time"] = min(summary["min_time"], total_time)
            
            # Calculate average
            if total_times:
                summary["average_time"] = sum(total_times) / len(total_times)
        
        return summary
    
    def clear_timing_data(self):
        """Clear all timing data."""
        self.timings = {}
        logger.debug("Cleared RAG timing data")

def time_rag_function(func):
    """Decorator to time RAG functions with logging.
    
    Args:
        func: Function to time
        
    Returns:
        Timed function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log start
        logger.debug(f"Starting {func.__name__}")
        start_time = time.time()
        
        # Call function
        result = func(*args, **kwargs)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log completion
        logger.debug(f"Completed {func.__name__} in {execution_time:.4f}s")
        
        # Add timing to result if it's a dictionary
        if isinstance(result, dict):
            result["execution_time"] = execution_time
        
        return result
    
    return wrapper