# core/rag/chain_builder.py

from typing import List, Dict, Any, Optional, Callable, Union
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.vector_store_factory import VectorStoreFactory
from core.rag.rag_pipeline import RAGPipeline

# Create a logger for this module
logger = get_module_logger("rag_chain_builder")

class RAGChainBuilder:
    """Builder for creating and configuring RAG pipelines with observability."""
    
    @staticmethod
    def build(
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        vector_store: Optional[Any] = None,
        store_type: Optional[str] = None,
        k_documents: Optional[int] = None,
        prompt_template: Optional[str] = None,
        observability_callbacks: Optional[List[Callable]] = None
    ) -> RAGPipeline:
        """Build a RAG pipeline with configuration.
        
        Args:
            api_key: OpenAI API key (default: from config)
            model_name: Model name (default: from config)
            temperature: Temperature (default: from config)
            vector_store: Vector store instance (created if not provided)
            store_type: Type of vector store to create if not provided
            k_documents: Number of documents to retrieve
            prompt_template: Custom prompt template
            observability_callbacks: List of callables for observability
            
        Returns:
            Configured RAG pipeline
        """
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name or config.llm.model_name,
            temperature=temperature if temperature is not None else config.llm.temperature,
            openai_api_key=api_key or config.llm.api_key
        )
        
        # Create or use vector store
        if not vector_store:
            vector_store = VectorStoreFactory.create_vector_store(store_type=store_type)
        
        # Get retriever from vector store
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k_documents or config.vector_store.similarity_top_k}
        )
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(
            llm=llm,
            retriever=retriever,
            prompt_template=prompt_template,
            k_documents=k_documents,
            observability_callbacks=observability_callbacks or []
        )
        
        logger.info(f"Built RAG pipeline with model {model_name or config.llm.model_name}")
        return rag_pipeline
    
    @staticmethod
    def add_default_observability(rag_pipeline: RAGPipeline) -> RAGPipeline:
        """Add default observability callbacks to a RAG pipeline.
        
        Args:
            rag_pipeline: Existing RAG pipeline
            
        Returns:
            RAG pipeline with observability
        """
        # Define logging callback
        def logging_callback(step: str, input: Any, output: Any):
            if step == "retrieval":
                doc_count = len(output) if isinstance(output, list) else 0
                logger.debug(f"Retrieved {doc_count} documents")
            elif step == "generation":
                output_sample = str(output)[:100] + "..." if output and len(str(output)) > 100 else output
                logger.debug(f"Generated output: {output_sample}")
            elif step == "end":
                if isinstance(output, dict) and "execution_time" in output:
                    logger.info(f"Query completed in {output['execution_time']:.2f}s")
        
        # Add callback to pipeline
        rag_pipeline.add_observability_callback(logging_callback)
        
        return rag_pipeline
    
    @staticmethod
    def create_prompt_template(template_type: str = "education") -> str:
        """Create a prompt template based on type.
        
        Args:
            template_type: Type of prompt template
            
        Returns:
            Prompt template string
        """
        if template_type == "education":
            return """You are a helpful AI assistant specializing in education and IEPs.
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context:
            {context}
            
            Question: {question}
            
            Helpful Answer:"""
        
        elif template_type == "concise":
            return """Answer the question based only on the following context:
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        
        else:
            # Default general template
            return """Use the following pieces of context to answer the question at the end.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""