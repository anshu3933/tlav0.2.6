# main.py

import os
import sys
import streamlit as st
from typing import Dict, Any, Optional, List
import time

from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.vector_store_factory import VectorStoreFactory
from core.llm.llm_client import LLMClient
from core.rag.chain_builder import RAGChainBuilder
from core.rag.rag_pipeline import RAGPipeline
from core.rag.observability import RagObservability
from ui.state_manager import state_manager
from fix_vector_store import verify_store_type

# Create a logger for this module
logger = get_module_logger("main")

def initialize_application() -> Dict[str, Any]:
    """Initialize the application components with improved error handling.
    
    Returns:
        Dictionary with initialized components
    """
    components = {}
    errors = []
    
    try:
        logger.info("Initializing application")
        
        # Verify environment
        if not check_environment():
            logger.error("Environment check failed")
            return {}
        
        # Verify vector store type
        store_type = os.environ.get("VECTOR_STORE_TYPE", "faiss")
        if not verify_store_type(store_type):
            logger.warning(f"Vector store type {store_type} verification failed, will be initialized")
        
        # Step 1: Initialize LLM client
        logger.debug("Initializing LLM client")
        try:
            llm_client = LLMClient()
            components["llm_client"] = llm_client
            logger.debug("LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM client: {str(e)}", exc_info=True)
            errors.append(f"LLM client initialization error: {str(e)}")
        
        # Step 2: Initialize vector store using factory
        logger.debug("Initializing vector store")
        try:
            embedding_provider = VectorStoreFactory.create_embeddings_provider()
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=store_type,
                embedding_provider=embedding_provider
            )
            components["vector_store"] = vector_store
            components["embedding_provider"] = embedding_provider
            logger.debug(f"Vector store initialized successfully: {type(vector_store).__name__}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
            errors.append(f"Vector store initialization error: {str(e)}")
        
        # Step 3: Initialize RAG observability
        logger.debug("Initializing RAG observability")
        try:
            rag_observability = RagObservability()
            components["rag_observability"] = rag_observability
            logger.debug("RAG observability initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG observability: {str(e)}", exc_info=True)
            errors.append(f"RAG observability initialization error: {str(e)}")
        
        # Step 4: Initialize RAG pipeline
        logger.debug("Initializing RAG pipeline")
        try:
            if "vector_store" in components and "llm_client" in components:
                # Get observability callback if available
                observability_callbacks = []
                if "rag_observability" in components:
                    observability_callbacks.append(components["rag_observability"].rag_step_callback())
                
                rag_pipeline = RAGPipeline(
                    llm=components["llm_client"],
                    retriever=components["vector_store"].as_retriever(),
                    observability_callbacks=observability_callbacks
                )
                components["rag_chain"] = rag_pipeline
                logger.debug("RAG pipeline initialized successfully")
            else:
                logger.warning("Skipping RAG pipeline initialization due to missing components")
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}", exc_info=True)
            errors.append(f"RAG pipeline initialization error: {str(e)}")
        
        # Step 5: Initialize pipeline components for IEP and lesson plans
        logger.debug("Initializing pipeline components")
        try:
            from core.pipelines.iep_pipeline import IEPGenerationPipeline
            from core.pipelines.lesson_plan_pipeline import LessonPlanGenerationPipeline
            
            if "llm_client" in components:
                # Create pipeline instances
                iep_pipeline = IEPGenerationPipeline(llm_client=components["llm_client"])
                lesson_plan_pipeline = LessonPlanGenerationPipeline(llm_client=components["llm_client"])
                components["iep_pipeline"] = iep_pipeline
                components["lesson_plan_pipeline"] = lesson_plan_pipeline
                logger.debug("Pipeline components initialized successfully")
            else:
                logger.warning("Skipping pipeline components initialization due to missing LLM client")
        except Exception as e:
            logger.error(f"Error initializing pipeline components: {str(e)}", exc_info=True)
            errors.append(f"Pipeline components initialization error: {str(e)}")
        
        # Update system state
        update_system_state(components)
        
        # Process existing documents if needed
        if "vector_store" in components:
            process_existing_data_files(components)
        
        # Log errors if any
        if errors:
            logger.warning(f"Application initialized with {len(errors)} errors")
            state_manager.set("initialization_errors", errors)
        else:
            logger.info("Application initialized successfully")
        
        return components
        
    except Exception as e:
        logger.error(f"Critical error initializing application: {str(e)}", exc_info=True)
        state_manager.add_error(f"Failed to initialize application: {str(e)}")
        return {}

def update_system_state(components: Dict[str, Any]):
    """Update system state with component status.
    
    Args:
        components: Dictionary with application components
    """
    system_state = {
        "llm_initialized": "llm_client" in components,
        "vector_store_initialized": "vector_store" in components and (
            hasattr(components["vector_store"], "_index_exists") and 
            components["vector_store"]._index_exists()
        ),
        "chain_initialized": "rag_chain" in components,
        "rag_observability_enabled": "rag_observability" in components
    }
    
    state_manager.update_system_state(**system_state)
    logger.debug(f"Updated system state: {system_state}")

def process_existing_data_files(app_components: Dict[str, Any]) -> None:
    """Process existing data files in the data directory on startup."""
    from langchain.schema import Document
    from core.document_processing.document_loader import DocumentLoader
    from core.document_processing.file_handler import FileHandler
    
    data_dir = config.document.data_dir
    logger.info(f"Checking for existing documents in {data_dir}")
    
    if not os.path.exists(data_dir):
        logger.debug(f"Data directory {data_dir} does not exist")
        return
    
    # Get vector store from components
    vector_store = app_components.get("vector_store")
    if not vector_store:
        logger.error("Vector store not initialized. Cannot process existing documents.")
        return
    
    # Initialize document loader
    document_loader = DocumentLoader()
    file_handler = FileHandler()
    
    # Process only if not already processed
    if not state_manager.get("documents_processed", False):
        files_processed = 0
        documents = []
        
        # List all files in data directory
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check file extension
            _, ext = os.path.splitext(filename)
            if ext.lower() not in ['.pdf', '.docx', '.txt', '.md', '.csv', '.json']:
                continue
            
            try:
                # Load document
                result = document_loader.load_single_document(file_path)
                
                if not result.success:
                    logger.warning(f"Error processing {filename}: {result.error_message}")
                    continue
                
                # Add to list
                document = result.document
                document.metadata["source"] = filename
                document.metadata["id"] = f"doc_{len(documents)}"
                documents.append(document)
                files_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
        
        # Add documents to vector store and state
        if documents:
            try:
                # Add to vector store
                success = vector_store.add_documents(documents)
                
                if success:
                    # Add to state
                    for doc in documents:
                        state_manager.append("documents", doc)
                    
                    state_manager.set("documents_processed", True)
                    logger.info(f"Processed {files_processed} existing documents on startup")
                    
                    # Update system state
                    state_manager.update_system_state(
                        vector_store_initialized=True
                    )
                else:
                    logger.error("Failed to add documents to vector store")
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}")

def check_environment() -> bool:
    """Check if the environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Check API key
    if not config.llm.api_key:
        logger.error("OpenAI API key not found in environment variables")
        state_manager.add_error("OpenAI API key not found. Please add it to your .env file.")
        return False
    
    # Check data directory
    if not os.path.exists(config.document.data_dir):
        try:
            os.makedirs(config.document.data_dir, exist_ok=True)
            logger.debug(f"Created data directory: {config.document.data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {str(e)}")
            state_manager.add_error(f"Failed to create data directory: {str(e)}")
            return False
    
    # Check index directory
    if not os.path.exists(config.vector_store.index_dir):
        try:
            os.makedirs(config.vector_store.index_dir, exist_ok=True)
            logger.debug(f"Created index directory: {config.vector_store.index_dir}")
        except Exception as e:
            logger.error(f"Failed to create index directory: {str(e)}")
            state_manager.add_error(f"Failed to create index directory: {str(e)}")
            return False
    
    return True

def load_app_components() -> Dict[str, Any]:
    """Initialize and load all application components with improved error handling.
    
    Returns:
        Dictionary with application components
    """
    # Track loading time for performance monitoring
    start_time = time.time()
    
    # Initialize the application
    components = initialize_application()
    
    # Log initialization time
    initialization_time = time.time() - start_time
    logger.info(f"Application components loaded in {initialization_time:.2f}s")
    
    # Store initialization time in state
    state_manager.set("initialization_time", initialization_time)
    
    # Check for initialization errors
    initialization_errors = state_manager.get("initialization_errors", [])
    if initialization_errors:
        for error in initialization_errors:
            state_manager.add_error(f"Initialization error: {error}")
    
    return components

def run_streamlit_app():
    """Entry point for the Streamlit application."""
    from ui.app import run_app
    run_app()

if __name__ == "__main__":
    run_streamlit_app()