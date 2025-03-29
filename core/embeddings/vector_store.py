# core/embeddings/vector_store.py

import os
import shutil
import time
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.app_config import config
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("vector_store")

class VectorStoreError(Exception):
    """Exception raised for vector store errors."""
    pass

class FAISSVectorStore:
    """Manages FAISS vector store operations using LangChain's implementation."""
    
    def __init__(self, 
                embedding_provider: Optional[Any] = None,
                index_dir: Optional[str] = None):
        """Initialize with components and directories.
        
        Args:
            embedding_provider: Provider for embeddings (default: OpenAIEmbeddings)
            index_dir: Directory to store the index
        """
        self.index_dir = index_dir or config.vector_store.index_dir
        
        # Use OpenAIEmbeddings as the default embedding provider
        self.embedding_provider = embedding_provider or OpenAIEmbeddings(
            model=config.vector_store.embedding_model
        )
        
        self.vectorstore = None
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        logger.debug(f"Initialized FAISS vector store with index directory: {self.index_dir}")
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> bool:
        """Build a FAISS index from documents.
        
        Args:
            documents: Documents to index
            force_rebuild: Whether to force rebuild even if index exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists
            if not force_rebuild and self._index_exists():
                logger.debug("Index already exists. Loading existing index.")
                return self.load_index()
            
            # Handle empty documents case
            if not documents or len(documents) == 0:
                logger.info("Creating empty FAISS index")
                
                # Create an empty vectorstore with a single placeholder document
                placeholder_doc = Document(
                    page_content="This is a placeholder document for empty index",
                    metadata={"source": "placeholder", "id": "placeholder_doc"}
                )
                
                # Extract texts and metadata separately for FAISS.from_texts
                texts = [placeholder_doc.page_content]
                metadatas = [placeholder_doc.metadata]
                
                # Create FAISS using from_texts instead of from_documents to avoid id attribute issue
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_provider,
                    metadatas=metadatas
                )
                
                # Save the empty index
                self.save_index()
                logger.info("Created empty FAISS index")
                return True
            
            logger.info(f"Building FAISS index with {len(documents)} documents")
            
            # Extract texts and metadata separately
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Get document content
                texts.append(doc.page_content)
                
                # Create metadata with ID if not present
                metadata = dict(doc.metadata) if doc.metadata else {}
                if 'id' not in metadata:
                    metadata['id'] = f"doc_{int(time.time())}_{i}"
                metadatas.append(metadata)
            
            # Create FAISS using from_texts instead of from_documents
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_provider,
                metadatas=metadatas
            )
            
            # Save the index
            self.save_index()
            
            logger.info(f"Successfully built FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            return False
    
    def save_index(self) -> bool:
        """Save the FAISS index to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                logger.error("No vectorstore to save")
                return False
            
            # Create a backup of existing index if it exists
            if self._index_exists():
                self._backup_index()
            
            # Save index using LangChain's built-in method
            self.vectorstore.save_local(self.index_dir)
            logger.info(f"Saved FAISS index to {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}", exc_info=True)
            return False
    
    def load_index(self) -> bool:
        """Load the FAISS index from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._index_exists():
                logger.error(f"FAISS index not found at {self.index_dir}")
                return False
            
            # Load index using LangChain's built-in method
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embedding_provider
            )
            
            logger.info(f"Loaded FAISS index from {self.index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}", exc_info=True)
            return False
    
    def _index_exists(self) -> bool:
        """Check if index exists on disk.
        
        Returns:
            True if index exists, False otherwise
        """
        return (
            os.path.exists(self.index_dir) and 
            os.path.exists(os.path.join(self.index_dir, "index.faiss")) and
            os.path.exists(os.path.join(self.index_dir, "index.pkl"))
        )
    
    def _backup_index(self) -> None:
        """Create a backup of the existing index."""
        try:
            # Create backup directory
            backup_dir = f"{self.index_dir}_backup_{int(time.time())}"
            
            # Copy index files
            shutil.copytree(self.index_dir, backup_dir)
            
            logger.debug(f"Created index backup at {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error creating index backup: {str(e)}")
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """Search for documents similar to the query.
        
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
                    raise VectorStoreError("No index available for search")
            
            # Use configurable k if not specified
            k = k or config.vector_store.similarity_top_k
            
            # Use the built-in similarity_search method
            results = self.vectorstore.similarity_search(query, k=k)
            
            logger.debug(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Search failed: {str(e)}")
    
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
            
            # Extract texts and metadata
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Get document content
                texts.append(doc.page_content)
                
                # Create metadata with ID if not present
                metadata = dict(doc.metadata) if doc.metadata else {}
                if 'id' not in metadata:
                    metadata['id'] = f"doc_{int(time.time())}_{i}"
                metadatas.append(metadata)
            
            # If we have an existing index
            if self.vectorstore:
                try:
                    # Try to add texts directly
                    self.vectorstore.add_texts(texts, metadatas=metadatas)
                    self.save_index()
                    logger.info(f"Added {len(documents)} documents to existing FAISS index")
                    return True
                except Exception as e:
                    logger.warning(f"Could not add to existing index: {str(e)}, rebuilding...")
                    
                    # Fall back to rebuilding the entire index
                    # Try to get existing documents first
                    existing_docs = []
                    try:
                        # This is a best effort attempt, may not work with all FAISS versions
                        if hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "_dict"):
                            for doc_id, doc_data in self.vectorstore.docstore._dict.items():
                                page_content = doc_data.get("page_content", "")
                                metadata = doc_data.get("metadata", {})
                                existing_docs.append(Document(page_content=page_content, metadata=metadata))
                    except Exception as ex:
                        logger.warning(f"Could not retrieve existing documents: {str(ex)}")
                    
                    # Combine existing and new documents
                    all_docs = existing_docs + documents
                    
                    # Rebuild with all documents
                    return self.build_index(all_docs, force_rebuild=True)
            else:
                # No existing vectorstore, build from scratch
                return self.build_index(documents)
        
        except Exception as e:
            logger.error(f"Error adding documents to FAISS index: {str(e)}", exc_info=True)
            return False
    
    def clear_index(self) -> bool:
        """Clear the index and remove all documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup before clearing
            if self._index_exists():
                self._backup_index()
            
            # Remove index files
            if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                os.remove(os.path.join(self.index_dir, "index.faiss"))
            
            if os.path.exists(os.path.join(self.index_dir, "index.pkl")):
                os.remove(os.path.join(self.index_dir, "index.pkl"))
            
            # Reset vectorstore
            self.vectorstore = None
            
            # Create a new empty index
            self.build_index([])
            
            logger.info("Cleared FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {str(e)}", exc_info=True)
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
                    raise VectorStoreError("No index available for retrieval")
            
            # Get the native retriever from LangChain
            retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs or {"k": config.vector_store.similarity_top_k}
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to create retriever: {str(e)}")