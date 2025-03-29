# core/embeddings/vertex_store.py

import os
import time
import uuid
from typing import List, Optional, Dict, Any, Callable
from langchain.schema import Document
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from config.app_config import config
from config.logging_config import get_module_logger
from core.embeddings.embedding_manager import TextChunkProcessor

# Create a logger for this module
logger = get_module_logger("vertex_store")

class VectorStoreError(Exception):
    """Exception raised for vector store errors."""
    pass

class VertexVectorStore:
    """Manages Vertex AI Vector Search operations with error handling."""
    
    def __init__(self, 
                project_id: Optional[str] = None,
                location: Optional[str] = None,
                index_id: Optional[str] = None,
                embedding_model: Optional[str] = None):
        """Initialize with Vertex AI configuration.
        
        Args:
            project_id: GCP project ID
            location: GCP location
            index_id: Vertex AI Vector Search index ID
            embedding_model: Name of the embedding model to use
        """
        self.project_id = project_id or os.environ.get("GCP_PROJECT", "your-project-id")
        self.location = location or os.environ.get("GCP_LOCATION", "us-central1")
        self.index_name = os.environ.get("VERTEX_INDEX_NAME", "educational-assistant-index")
        self.index_id = index_id or self._get_or_create_index()
        self.embedding_model = embedding_model or "textembedding-gecko@latest"
        self.dimension = 768  # Gecko model dimension
        self.chunk_processor = TextChunkProcessor()
        self.vector_search = None
        
        # Initialize embeddings
        self.embeddings = VertexAIEmbeddings(model_name=self.embedding_model)
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        
        logger.debug(f"Initialized Vertex AI Vector Search with index: {self.index_name}")
        
        # Ensure index exists or create it
        self._ensure_index_exists()
    
    def _get_or_create_index(self) -> str:
        """Get existing index ID or create a new one."""
        try:
            # List existing indexes
            from google.cloud import aiplatform_v1
            
            client = aiplatform_v1.IndexServiceClient()
            parent = f"projects/{self.project_id}/locations/{self.location}"
            indexes = client.list_indexes(parent=parent)
            
            # Check if our index exists
            for index in indexes:
                if index.display_name == self.index_name:
                    logger.info(f"Found existing index: {index.name}")
                    # Extract just the ID from full name
                    return index.name.split('/')[-1]
            
            # No existing index, create one
            return self._create_new_index()
        except Exception as e:
            logger.error(f"Error getting index: {str(e)}")
            return None
    
    def _create_new_index(self) -> str:
        """Create a new Vector Search index."""
        try:
            logger.info(f"Creating new Vertex AI index: {self.index_name}")
            
            index = aiplatform.MatchingEngineIndex.create(
                display_name=self.index_name,
                dimensions=self.dimension, 
                approximate_neighbors_count=150,
                distance_measure_type="DOT_PRODUCT_DISTANCE"
            )
            
            # Wait for index to be created (this can take a while)
            logger.info(f"Waiting for index creation to complete")
            index.wait()
            
            return index.name.split('/')[-1]
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return None
    
    def _ensure_index_exists(self) -> bool:
        """Ensure index exists and is ready for use."""
        if not self.index_id:
            logger.warning("No index ID available. Attempting to create index.")
            self.index_id = self._create_new_index()
            
        if not self.index_id:
            logger.error("Failed to get or create index")
            return False
            
        return True
    
    def _index_exists(self) -> bool:
        """Check if index exists and is available."""
        return self.index_id is not None
    
    def load_index(self) -> bool:
        """Load the index for querying.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._index_exists():
                logger.error("Index not found")
                return False
            
            # Nothing special needed for loading - index exists in the cloud
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            return False
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> bool:
        """Build or update the Vertex AI index from documents.
        
        Args:
            documents: Documents to index
            force_rebuild: Whether to force rebuild (not used for Vertex)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check index exists
            if not self._ensure_index_exists():
                return False
            
            # Just add documents - Vertex handles it all
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error building Vertex index: {str(e)}", exc_info=True)
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the index.
        
        Args:
            documents: Documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.debug("No documents to add")
                return True
            
            # Process documents into chunks
            chunked_docs = self.chunk_processor.split_documents(documents)
            
            if not chunked_docs:
                logger.warning("No document chunks to add")
                return False
            
            # Import here to avoid circular imports
            from langchain_google_vertexai import VertexAIVector
            
            # Create unique IDs for each document
            ids = [str(uuid.uuid4()) for _ in range(len(chunked_docs))]
            
            # Add documents to index
            VertexAIVector.from_documents(
                documents=chunked_docs,
                embedding=self.embeddings,
                index_name=self.index_name,
                project_id=self.project_id,
                location=self.location,
                ids=ids
            )
            
            logger.info(f"Added {len(chunked_docs)} document chunks to Vertex AI index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Vertex index: {str(e)}", exc_info=True)
            return False
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """Search the index for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            VectorStoreError: If search fails
        """
        try:
            if not self._index_exists():
                raise VectorStoreError("No index available for search")
            
            # Use configurable k if not specified
            k = k or config.vector_store.similarity_top_k
            
            # Import here to avoid circular imports
            from langchain_google_vertexai import VertexAIVector
            
            # Create vector store for querying
            vector_store = VertexAIVector(
                embedding=self.embeddings,
                index_name=self.index_name,
                project_id=self.project_id,
                location=self.location
            )
            
            # Search documents
            results = vector_store.similarity_search(query, k=k)
            
            logger.debug(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Vertex index: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Callable:
        """Get a retriever function for the vector store.
        
        Args:
            search_kwargs: Search parameters
            
        Returns:
            Retriever function
            
        Raises:
            VectorStoreError: If retriever creation fails
        """
        try:
            if not self._index_exists():
                raise VectorStoreError("No index available for retrieval")
            
            # Create search parameters with defaults
            search_kwargs = search_kwargs or {
                "k": config.vector_store.similarity_top_k
            }
            
            # Create retriever function
            def retriever(query: str) -> List[Document]:
                return self.search(query, k=search_kwargs.get("k"))
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to create retriever: {str(e)}")
    
    def clear_index(self) -> bool:
        """Clear the index and remove all documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index_id:
                logger.warning("No index to clear")
                return True
            
            # Delete the index
            from google.cloud import aiplatform_v1
            
            client = aiplatform_v1.IndexServiceClient()
            name = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
            
            # Create delete operation
            operation = client.delete_index(name=name)
            operation.result()  # Wait for operation to complete
            
            # Reset index ID
            self.index_id = None
            
            # Create a new empty index
            self.index_id = self._create_new_index()
            
            logger.info("Cleared Vertex AI index")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}", exc_info=True)
            return False