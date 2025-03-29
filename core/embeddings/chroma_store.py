# core/embeddings/chroma_store.py

import os
import time
from typing import List, Optional, Dict, Any, Callable
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.app_config import config
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("chroma_store")

class VectorStoreError(Exception):
    """Exception raised for vector store errors."""
    pass

class ChromaVectorStore:
    """Manages ChromaDB vector store operations using LangChain's implementation."""
    
    def __init__(self, 
                embedding_provider: Optional[Any] = None,
                persist_directory: Optional[str] = None):
        """Initialize with components and directories.
        
        Args:
            embedding_provider: Provider for embeddings (default: OpenAIEmbeddings)
            persist_directory: Directory to store the database
        """
        self.persist_directory = persist_directory or os.path.join(config.vector_store.index_dir, "chroma_db")
        
        # Use OpenAIEmbeddings as the default embedding provider
        self.embedding_provider = embedding_provider or OpenAIEmbeddings(
            model=config.vector_store.embedding_model
        )
        
        self.collection_name = "documents"
        self.vectorstore = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.debug(f"Initialized ChromaDB vector store with directory: {self.persist_directory}")
    
    def _index_exists(self) -> bool:
        """Check if index exists on disk.
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            # Try to import ChromaDB
            import chromadb
            
            # Check if the directory exists and is not empty
            if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
                return False
            
            # Try to connect to ChromaDB and check if the collection exists
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get collection names - this handles different API versions
            try:
                collections = client.list_collections()
                if isinstance(collections, list):
                    # Newer API that returns collection objects
                    if isinstance(collections[0], str) if collections else False:
                        return self.collection_name in collections
                    else:
                        return self.collection_name in [c.name for c in collections]
                return False
            except Exception:
                # If the above fails, try the alternative API
                return False
            
        except Exception as e:
            logger.error(f"Error checking if ChromaDB collection exists: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """Load the ChromaDB index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection exists on disk
            if not self._index_exists():
                logger.warning(f"ChromaDB collection not found: {self.collection_name}")
                return False
            
            # Create Chroma instance
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_provider,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Loaded ChromaDB collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ChromaDB: {str(e)}", exc_info=True)
            return False
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> bool:
        """Build a ChromaDB index from documents.
        
        Args:
            documents: Documents to index
            force_rebuild: Whether to force rebuild even if index exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists
            if not force_rebuild and self._index_exists():
                logger.debug("ChromaDB collection already exists. Loading existing collection.")
                return self.load_index()
            
            # Force delete existing collection if requested
            if force_rebuild and self._index_exists():
                self.clear_index()
            
            # Handle empty documents case
            if not documents or len(documents) == 0:
                logger.info("Creating empty ChromaDB collection")
                
                # Create an empty vectorstore with a single placeholder document
                placeholder_doc = Document(
                    page_content="This is a placeholder document for empty index",
                    metadata={"source": "placeholder", "id": "placeholder_doc"}
                )
                
                self.vectorstore = Chroma.from_documents(
                    [placeholder_doc],
                    embedding=self.embedding_provider,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                
                # Make sure changes are persisted
                self.vectorstore.persist()
                
                logger.info("Created empty ChromaDB collection")
                return True
            
            logger.info(f"Building ChromaDB index with {len(documents)} documents")
            
            # Make sure documents have IDs in metadata
            docs_with_ids = []
            for i, doc in enumerate(documents):
                # Create a new document with the same content and metadata
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata) if doc.metadata else {}
                )
                # Make sure the document has an ID in its metadata
                if 'id' not in new_doc.metadata:
                    new_doc.metadata['id'] = f"doc_{int(time.time())}_{i}"
                docs_with_ids.append(new_doc)
            
            # Create ChromaDB index
            self.vectorstore = Chroma.from_documents(
                docs_with_ids,
                embedding=self.embedding_provider,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                ids=[doc.metadata['id'] for doc in docs_with_ids]
            )
            
            # Make sure changes are persisted
            self.vectorstore.persist()
            
            logger.info(f"Successfully built ChromaDB index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error building ChromaDB index: {str(e)}", exc_info=True)
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store.
        
        Args:
            documents: Documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return True
            
            # Load existing index or create a new one
            if not self.vectorstore:
                if not self.load_index() and not self.build_index([]):
                    logger.error("Failed to load or create index for adding documents")
                    return False
            
            # Generate IDs for the documents if they don't have them in metadata
            docs_with_ids = []
            for i, doc in enumerate(documents):
                # Create a new document with the same content and metadata
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=dict(doc.metadata) if doc.metadata else {}
                )
                # Make sure the document has an ID in its metadata
                if 'id' not in new_doc.metadata:
                    new_doc.metadata['id'] = f"doc_{int(time.time())}_{i}"
                docs_with_ids.append(new_doc)
            
            # Add documents with explicit IDs
            ids = [doc.metadata['id'] for doc in docs_with_ids]
            self.vectorstore.add_documents(docs_with_ids, ids=ids)
            
            # Remove the persist call - newer Chroma versions persist automatically
            # No need to call self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}", exc_info=True)
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
            if not self.vectorstore:
                if not self.load_index():
                    raise VectorStoreError("No ChromaDB available for search")
            
            # Use configurable k if not specified
            k = k or config.vector_store.similarity_top_k
            
            # Search documents using LangChain's implementation
            results = self.vectorstore.similarity_search(query, k=k)
            
            logger.debug(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def clear_index(self) -> bool:
        """Clear the index and remove all documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection
            import chromadb
            
            try:
                client = chromadb.PersistentClient(path=self.persist_directory)
                
                # Get collection names
                collections = client.list_collections()
                collection_exists = False
                
                if isinstance(collections, list):
                    # Check if collection exists
                    if isinstance(collections[0], str) if collections else False:
                        collection_exists = self.collection_name in collections
                    else:
                        collection_exists = self.collection_name in [c.name for c in collections]
                
                # Delete if exists
                if collection_exists:
                    client.delete_collection(self.collection_name)
                    logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Error deleting collection: {str(e)}")
            
            # Reset vectorstore reference
            self.vectorstore = None
            
            # Create a new empty collection
            return self.build_index([])
            
        except Exception as e:
            logger.error(f"Error clearing ChromaDB index: {str(e)}", exc_info=True)
            return False
    
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Get a retriever for the vector store.
        
        Args:
            search_kwargs: Search parameters
            
        Returns:
            Retriever object
            
        Raises:
            VectorStoreError: If retriever creation fails
        """
        try:
            if not self.vectorstore:
                if not self.load_index():
                    raise VectorStoreError("No ChromaDB available for retrieval")
            
            # Get the native retriever from LangChain
            retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs or {"k": config.vector_store.similarity_top_k}
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to create retriever: {str(e)}")