# core/embeddings/text_chunker.py

from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from config.app_config import config
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("text_chunker")

class TextChunkProcessor:
    """Handles text chunking for embeddings with multiple strategies."""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 chunking_strategy: str = "recursive",
                 embedding_provider: Optional[Any] = None):
        """Initialize with chunking parameters.
        
        Args:
            chunk_size: Size of each chunk (default: from config)
            chunk_overlap: Overlap between chunks (default: from config)
            chunking_strategy: Chunking strategy (recursive, semantic)
            embedding_provider: Provider for embeddings in semantic chunking
        """
        self.chunk_size = chunk_size or config.vector_store.chunk_size
        self.chunk_overlap = chunk_overlap or config.vector_store.chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.embedding_provider = embedding_provider
        
        # Initialize chunkers
        self._init_chunkers()
        
        logger.debug(f"Initialized text chunker with strategy={chunking_strategy}, "
                    f"size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def _init_chunkers(self):
        """Initialize text splitters based on strategy."""
        # Initialize recursive chunker (default)
        self.recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize semantic chunker if specified
        if self.chunking_strategy == "semantic":
            # Create embeddings provider if not provided
            if not self.embedding_provider:
                self.embedding_provider = OpenAIEmbeddings(
                    model=config.vector_store.embedding_model
                )
            
            # Create semantic chunker
            self.semantic_chunker = SemanticChunker(
                self.embedding_provider,
                breakpoint_threshold_modifier=0.3
            )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using the selected strategy.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        try:
            # Use semantic chunking if specified
            if self.chunking_strategy == "semantic" and hasattr(self, "semantic_chunker"):
                chunks = self.semantic_chunker.split_text(text)
            else:
                # Use recursive chunking by default
                chunks = self.recursive_chunker.split_text(text)
            
            logger.debug(f"Split text into {len(chunks)} chunks using {self.chunking_strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}", exc_info=True)
            # Fallback to simple chunking
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """Simple fallback splitting method when main strategy fails.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Simple paragraph-based splitting
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Add separator if not first paragraph in chunk
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"Fallback split text into {len(chunks)} chunks")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preservation.
        
        Args:
            documents: Documents to split
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            # Skip empty documents
            if not doc.page_content.strip():
                continue
                
            # Split the document text
            chunks = self.split_text(doc.page_content)
            
            # Create a new document for each chunk
            for i, chunk in enumerate(chunks):
                # Create a copy of the metadata
                metadata = dict(doc.metadata) if doc.metadata else {}
                
                # Add chunking metadata
                metadata.update({
                    "chunk": i,
                    "chunk_size": self.chunk_size,
                    "chunk_strategy": self.chunking_strategy,
                    "total_chunks": len(chunks)
                })
                
                # Create new document
                chunked_doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                chunked_docs.append(chunked_doc)
        
        logger.debug(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs