# core/llm/dspy/adapter.py

"""Adapter for DSPy framework integration for structured reasoning."""

import os
from typing import Dict, List, Any, Optional, Union, Callable, Type
import time
import importlib
from config.logging_config import get_module_logger
from core.llm.bridge import LLMResponse, LLMRequest, ai_adapter, ModelProvider

# Create a logger for this module
logger = get_module_logger("dspy_adapter")

class DSPyAdapter:
    """Adapter for DSPy framework integration."""
    
    def __init__(self, 
                 model_provider: ModelProvider = ModelProvider.OPENAI,
                 model_name: str = "gpt-4o"):
        """Initialize with model provider and name.
        
        Args:
            model_provider: Model provider
            model_name: Model name
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.dspy_available = self._check_dspy_available()
        
        if self.dspy_available:
            self._initialize_dspy()
        else:
            logger.warning("DSPy not available. Some features will be limited.")
        
        logger.debug(f"Initialized DSPy adapter with {model_provider.value} {model_name}")
    
    def _check_dspy_available(self) -> bool:
        """Check if DSPy is available.
        
        Returns:
            True if DSPy is available
        """
        try:
            importlib.import_module("dspy")
            return True
        except ImportError:
            return False
    
    def _initialize_dspy(self) -> None:
        """Initialize DSPy components."""
        try:
            import dspy
            from dspy.teleprompt import Teleprompt
            
            # Create DSPy OpenAI LM
            if self.model_provider == ModelProvider.OPENAI:
                from dspy.openai import OpenAI
                
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OpenAI API key not found in environment variables")
                    self.dspy_lm = None
                    return
                
                self.dspy_lm = OpenAI(model=self.model_name, api_key=api_key)
                dspy.settings.configure(lm=self.dspy_lm)
                
            # Create DSPy Anthropic LM
            elif self.model_provider == ModelProvider.ANTHROPIC:
                from dspy.anthropic import Claude
                
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.error("Anthropic API key not found in environment variables")
                    self.dspy_lm = None
                    return
                
                self.dspy_lm = Claude(model=self.model_name, api_key=api_key)
                dspy.settings.configure(lm=self.dspy_lm)
            
            # Create other components
            self.teleprompt = Teleprompt()
            
            logger.debug("Initialized DSPy components")
            
        except ImportError as e:
            logger.error(f"Error importing DSPy component: {str(e)}")
            self.dspy_lm = None
        except Exception as e:
            logger.error(f"Error initializing DSPy: {str(e)}")
            self.dspy_lm = None
    
    def create_module(self, signature_class: Type) -> Any:
        """Create a DSPy module from a signature class.
        
        Args:
            signature_class: DSPy signature class
            
        Returns:
            DSPy module
        """
        if not self.dspy_available or not self.dspy_lm:
            logger.error("DSPy not available or not initialized")
            return None
        
        try:
            import dspy
            return dspy.Module(signature_class)
        except Exception as e:
            logger.error(f"Error creating DSPy module: {str(e)}")
            return None
    
    def run_module(self, module: Any, **kwargs) -> Dict[str, Any]:
        """Run a DSPy module with arguments.
        
        Args:
            module: DSPy module
            **kwargs: Arguments for the module
            
        Returns:
            Module outputs
        """
        if not self.dspy_available or not self.dspy_lm:
            logger.error("DSPy not available or not initialized")
            return {}
        
        try:
            # Run module
            result = module(**kwargs)
            
            # Convert to dictionary
            if hasattr(result, "__dict__"):
                return {k: v for k, v in result.__dict__.items() 
                       if not k.startswith("_")}
            else:
                return {"result": str(result)}
        except Exception as e:
            logger.error(f"Error running DSPy module: {str(e)}")
            return {"error": str(e)}
    
    def optimize_prompt(self, module: Any, trainset: List[Dict[str, Any]], metric: Optional[Callable] = None) -> Any:
        """Optimize prompts using Teleprompt.
        
        Args:
            module: DSPy module to optimize
            trainset: Training examples
            metric: Optional evaluation metric
            
        Returns:
            Optimized module
        """
        if not self.dspy_available or not self.dspy_lm or not hasattr(self, "teleprompt"):
            logger.error("DSPy Teleprompt not available or not initialized")
            return module
        
        try:
            import dspy
            
            # Create default metric if not provided
            if metric is None:
                def default_metric(example, pred):
                    # Simple accuracy metric
                    return 1.0 if pred == example.expected_output else 0.0
                
                metric = default_metric
            
            # Convert trainset to DSPy examples
            dspy_examples = []
            for item in trainset:
                inputs = {k: v for k, v in item.items() if k != "expected_output"}
                expected = item.get("expected_output", "")
                
                example = dspy.Example(inputs=inputs, expected_output=expected)
                dspy_examples.append(example)
            
            # Optimize prompts
            optimized_module = self.teleprompt.optimize(
                module=module,
                trainset=dspy_examples,
                metric=metric
            )
            
            return optimized_module
            
        except Exception as e:
            logger.error(f"Error optimizing prompts: {str(e)}")
            return module
    
    def chain_of_thought(self, prompt: str, **kwargs) -> str:
        """Generate chain of thought reasoning.
        
        Args:
            prompt: Prompt to reason about
            **kwargs: Additional parameters
            
        Returns:
            Reasoning response
        """
        if not self.dspy_available:
            # Fallback to standard AI adapter
            system_prompt = "Please think step by step to solve this problem."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = ai_adapter.chat(messages, **kwargs)
            return response.text
        
        try:
            import dspy
            
            # Create chain of thought signature
            class ChainOfThought(dspy.Signature):
                """Generate chain of thought reasoning."""
                question = dspy.InputField()
                reasoning = dspy.OutputField(desc="Step-by-step reasoning to solve the problem")
                answer = dspy.OutputField(desc="The final answer")
            
            # Create module
            cot_module = self.create_module(ChainOfThought)
            
            # Run module
            result = self.run_module(cot_module, question=prompt)
            
            # Format response
            response = f"Reasoning: {result.get('reasoning', '')}\n\nAnswer: {result.get('answer', '')}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chain of thought: {str(e)}")
            
            # Fallback to standard AI adapter
            system_prompt = "Please think step by step to solve this problem."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = ai_adapter.chat(messages, **kwargs)
            return response.text
    
    def retrieve_and_generate(self, 
                             query: str, 
                             documents: List[Dict[str, str]],
                             **kwargs) -> Dict[str, Any]:
        """Retrieve relevant information and generate a response.
        
        Args:
            query: User query
            documents: List of documents with content and metadata
            **kwargs: Additional parameters
            
        Returns:
            Response with answer and sources
        """
        if not self.dspy_available:
            # Fallback to standard approach
            context = "\n\n".join([doc.get("content", "") for doc in documents])
            
            system_prompt = """
            You are an educational assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information."
            Cite sources by their ID when possible.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = ai_adapter.chat(messages, **kwargs)
            
            return {
                "answer": response.text,
                "sources": [doc.get("id", i) for i, doc in enumerate(documents)]
            }
        
        try:
            import dspy
            from dspy.retrieve import BM25Retriever
            
            # Create RAG signature
            class RAG(dspy.Module):
                """Retrieve and generate a response."""
                def __init__(self, retriever):
                    super().__init__()
                    self.retriever = retriever
                    self.generate = dspy.ChainOfThought("context, question -> answer")
                
                def forward(self, question):
                    context = self.retriever(question)
                    answer = self.generate(context=context, question=question)
                    return dspy.Prediction(answer=answer.answer, context=context)
            
            # Create corpus
            corpus = [dspy.Document(content=doc.get("content", ""), 
                                 id=doc.get("id", str(i))) 
                   for i, doc in enumerate(documents)]
            
            # Create retriever
            retriever = BM25Retriever(corpus)
            
            # Create RAG module
            rag_module = RAG(retriever)
            
            # Run RAG
            result = rag_module(question=query)
            
            # Format response
            return {
                "answer": result.answer,
                "sources": [s.id for s in result.context]
            }
            
        except Exception as e:
            logger.error(f"Error in retrieve and generate: {str(e)}")
            
            # Fallback to standard approach
            context = "\n\n".join([doc.get("content", "") for doc in documents])
            
            system_prompt = """
            You are an educational assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say "I don't have enough information."
            Cite sources by their ID when possible.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = ai_adapter.chat(messages, **kwargs)
            
            return {
                "answer": response.text,
                "sources": [doc.get("id", i) for i, doc in enumerate(documents)]
            }

# Create a singleton instance
try:
    dspy_adapter = DSPyAdapter()
except Exception as e:
    logger.error(f"Error creating DSPy adapter: {str(e)}")
    dspy_adapter = None