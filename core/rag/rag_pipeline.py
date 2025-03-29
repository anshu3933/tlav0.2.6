# core/rag/rag_pipeline.py

from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from langchain.schema import Document
from config.app_config import config
from config.logging_config import get_module_logger
from core.llm.llm_client import LLMClient

# Create a logger for this module
logger = get_module_logger("rag_pipeline")

class RAGPipeline:
    """RAG pipeline with standardized components and observability."""
    
    def __init__(self, 
                 llm: Optional[Any] = None,
                 retriever: Optional[Any] = None,
                 prompt_template: Optional[str] = None,
                 k_documents: int = None,
                 observability_callbacks: List[Callable] = None):
        """Initialize with components.
        
        Args:
            llm: LLM to use for generation
            retriever: Document retriever function
            prompt_template: Prompt template for RAG
            k_documents: Number of documents to retrieve
            observability_callbacks: Callbacks for pipeline observability
        """
        # Initialize LLM
        self.llm = llm
        
        # Set retriever
        self.retriever = retriever
        
        # Set prompt template
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        # Set document count
        self.k_documents = k_documents or config.vector_store.similarity_top_k
        
        # Initialize observability callbacks
        self.observability_callbacks = observability_callbacks or []
        
        # Define format docs function
        self.format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
        
        # Build the RAG chain
        self._build_rag_chain()
        
        logger.debug("Initialized RAG pipeline")
    
    def _get_default_prompt(self) -> str:
        """Get the default RAG prompt template."""
        return """You are a helpful AI assistant specializing in education.
        Use the following pieces of context to answer the question at the end.
        If you find relevant information in the context, use it to provide a detailed answer.
        Even if the context is limited, try to provide a helpful response using your general knowledge.
        Only say "I don't know" if the question is completely unrelated to the context or impossible to answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    
    def _build_rag_chain(self):
        """Build the RAG chain using direct function calls instead of pipe operators."""
        # Create a simple callable chain instead of using the pipe operator
        # This avoids compatibility issues with different LangChain versions
        
        def chain_runner(query):
            # Step 1: Retrieve context and documents
            context, docs = self._retrieval_step(query)
            
            # Step 2: Format prompt with context and question
            prompt = self._prompt_step({"context": context, "question": query})
            
            # Step 3: Generate response
            response = self._generation_step(prompt)
            
            return response
        
        self.rag_chain = chain_runner
    
    def _retrieval_step(self, query: str) -> Tuple[str, List[Document]]:
        """Retrieval step with observability.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (formatted context, source documents)
        """
        # Check if retriever is available
        if not self.retriever:
            logger.warning("No retriever available for RAG pipeline")
            return "No context available.", []
        
        try:
            # Retrieve documents using the correct method based on retriever type
            # VectorStoreRetriever has a get_relevant_documents method, not callable directly
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(query)
            else:
                # Fallback for other retriever types
                docs = self.retriever(query)
            
            # Log retrieved documents
            logger.debug(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            
            # Call observability callbacks for retrieval step
            for callback in self.observability_callbacks:
                callback(step="retrieval", input=query, output=docs)
            
            # Format documents
            formatted_context = self.format_docs(docs)
            
            return formatted_context, docs
            
        except Exception as e:
            logger.error(f"Error in retrieval step: {str(e)}", exc_info=True)
            return "Error retrieving context.", []
    
    def _prompt_step(self, inputs: Dict[str, Any]) -> str:
        """Prompt formatting step with observability.
        
        Args:
            inputs: Input dictionary with context and question
            
        Returns:
            Formatted prompt
        """
        try:
            # Extract context and question
            context = inputs["context"]
            question = inputs["question"]
            
            # Format prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Call observability callbacks for prompt step
            for callback in self.observability_callbacks:
                callback(step="prompt", input=inputs, output=prompt)
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error in prompt step: {str(e)}", exc_info=True)
            return f"Error formatting prompt. Question: {inputs.get('question', 'Unknown')}"
    
    def _generation_step(self, prompt: str) -> str:
        """Generation step with observability.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated response
        """
        try:
            # Generate response - handle different LLM types
            if hasattr(self.llm, 'invoke'):
                # Native LangChain ChatModel
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
            elif hasattr(self.llm, 'chat_completion'):
                # Custom LLMClient
                response = self.llm.chat_completion(
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response["content"] if isinstance(response, dict) and "content" in response else str(response)
            else:
                # Fallback
                logger.warning("Unknown LLM type, attempting to call directly")
                response = self.llm(prompt)
                content = str(response)
            
            # Call observability callbacks for generation step
            for callback in self.observability_callbacks:
                callback(step="generation", input=prompt, output=content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error in generation step: {str(e)}", exc_info=True)
            return "Sorry, I encountered an error while generating a response."
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline on a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and additional info
        """
        try:
            # Log the query
            logger.info(f"Processing query: {query[:50]}...")
            
            # Call observability callbacks for run start
            for callback in self.observability_callbacks:
                callback(step="start", input=query, output=None)
            
            # Start timers and metrics
            start_time = __import__('time').time()
            
            # Run the retrieval step separately to get documents
            context, source_docs = self._retrieval_step(query)
            
            # Run the chain
            result = self.rag_chain(query)
            
            # Calculate execution time
            execution_time = __import__('time').time() - start_time
            
            # Create response
            response = {
                "result": result,
                "source_documents": source_docs,
                "execution_time": execution_time,
                "metadata": {
                    "query": query,
                    "num_docs": len(source_docs) if source_docs else 0
                }
            }
            
            # Call observability callbacks for run end
            for callback in self.observability_callbacks:
                callback(step="end", input=query, output=response)
            
            # Log completion
            logger.info(f"Completed query in {execution_time:.2f}s with {len(source_docs)} documents")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            return {
                "result": f"Error processing query: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def add_observability_callback(self, callback: Callable):
        """Add an observability callback to the pipeline.
        
        Args:
            callback: Callback function that accepts step, input, and output
        """
        self.observability_callbacks.append(callback)
        logger.debug(f"Added observability callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
    
    def set_retriever(self, retriever: Any):
        """Set the retriever for the pipeline.
        
        Args:
            retriever: Retriever object
        """
        self.retriever = retriever
        logger.debug("Updated retriever in RAG pipeline")
    
    def set_llm(self, llm: Any):
        """Set the LLM for the pipeline.
        
        Args:
            llm: Language model
        """
        self.llm = llm
        logger.debug(f"Updated LLM in RAG pipeline: {type(llm).__name__}")