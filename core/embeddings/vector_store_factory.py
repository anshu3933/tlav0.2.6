# core/embeddings/vector_store_factory.py

import os
from typing import Optional, Any, Dict
from langchain_openai import OpenAIEmbeddings
from config.logging_config import get_module_logger
from config.app_config import config

# Create a logger for this module
logger = get_module_logger("vector_store_factory")

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create_vector_store(
        store_type: str = None, 
        embedding_provider: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create a vector store instance based on type.
        
        Args:
            store_type: Type of vector store to create ('faiss', 'chroma', 'vertex')
            embedding_provider: Provider for embeddings (default: OpenAIEmbeddings)
            **kwargs: Additional arguments for the vector store
            
        Returns:
            Vector store instance
        """
        # Default to environment variable or configuration
        store_type = store_type or os.environ.get("VECTOR_STORE_TYPE", "faiss")
        
        # Setup default embedding provider if not provided
        if embedding_provider is None:
            embedding_provider = OpenAIEmbeddings(
                model=config.vector_store.embedding_model
            )
        
        try:
            logger.debug(f"Creating vector store of type: {store_type}")
            
            if store_type.lower() == "faiss":
                from core.embeddings.vector_store import FAISSVectorStore
                store = FAISSVectorStore(embedding_provider=embedding_provider, **kwargs)
                
                # Load or create index on initialization
                if not store._index_exists():
                    logger.info("Creating empty FAISS index during initialization")
                    store.build_index([])
                else:
                    store.load_index()
                return store
                
            elif store_type.lower() == "chroma":
                from core.embeddings.chroma_store import ChromaVectorStore
                store = ChromaVectorStore(embedding_provider=embedding_provider, **kwargs)
                
                # Load or create index on initialization
                if not store._index_exists():
                    logger.info("Creating empty ChromaDB collection during initialization")
                    store.build_index([])
                else:
                    store.load_index()
                return store
                
            elif store_type.lower() == "vertex":
                try:
                    from core.embeddings.vertex_store import VertexVectorStore
                    store = VertexVectorStore(**kwargs)
                    
                    # Ensure index exists on initialization
                    if not store._index_exists():
                        logger.info("Creating empty Vertex AI index during initialization")
                        store.build_index([])
                    return store
                except ImportError:
                    logger.warning("Vertex AI dependencies not installed. Falling back to FAISS.")
                    from core.embeddings.vector_store import FAISSVectorStore
                    return FAISSVectorStore(embedding_provider=embedding_provider, **kwargs)
                
            else:
                logger.warning(f"Unknown vector store type: {store_type}. Using FAISS.")
                from core.embeddings.vector_store import FAISSVectorStore
                store = FAISSVectorStore(embedding_provider=embedding_provider, **kwargs)
                
                # Load or create index on initialization
                if not store._index_exists():
                    logger.info("Creating empty FAISS index during initialization")
                    store.build_index([])
                else:
                    store.load_index()
                return store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
            # Fallback to FAISS as a last resort
            from core.embeddings.vector_store import FAISSVectorStore
            logger.warning(f"Falling back to FAISS vector store due to error")
            store = FAISSVectorStore(embedding_provider=embedding_provider, **kwargs)
            store.build_index([])
            return store

    @staticmethod
    def create_embeddings_provider(model: str = None) -> Any:
        """Create an embeddings provider.
        
        Args:
            model: Model name (defaults to config)
            
        Returns:
            Embeddings provider
        """
        try:
            # Use config model if not specified
            model = model or config.vector_store.embedding_model
            
            # Create OpenAI embeddings provider
            return OpenAIEmbeddings(model=model)
        except Exception as e:
            logger.error(f"Error creating embeddings provider: {str(e)}", exc_info=True)
            # Fallback to default model
            return OpenAIEmbeddings(model="text-embedding-ada-002")