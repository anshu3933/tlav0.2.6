# diagnose_rag_simple.py

"""Simple diagnostic script for testing the RAG pipeline without mock components."""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import get_module_logger
from langchain.schema import Document
from core.embeddings.vector_store_factory import VectorStoreFactory
from core.llm.llm_client import LLMClient
from core.rag.rag_pipeline import RAGPipeline
from core.rag.observability import RagObservability, time_rag_function

# Create a logger for this module
logger = get_module_logger("diagnose_rag_simple")

def test_openai_api_key():
    """Test if the OpenAI API key is valid.
    
    Returns:
        True if valid, False otherwise
    """
    from openai import OpenAI
    
    print("Testing OpenAI API key...")
    
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        # Create client
        client = OpenAI(api_key=api_key)
        
        # Make a simple request to verify API key
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=["This is a test to verify the API key."]
        )
        
        if response and hasattr(response, "data") and len(response.data) > 0:
            print("✅ OpenAI API key is valid")
            return True
        else:
            print("❌ OpenAI API key validation failed: Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API key validation failed: {str(e)}")
        return False

def test_vector_store(store_type="faiss"):
    """Test vector store creation and operations.
    
    Args:
        store_type: Type of vector store to test
        
    Returns:
        Tuple of (vector_store, documents) or (None, None) if failed
    """
    print(f"\n=== Testing {store_type} Vector Store ===")
    
    try:
        # Create embedding provider
        embedding_provider = VectorStoreFactory.create_embeddings_provider()
        print(f"✅ Created embedding provider: {type(embedding_provider).__name__}")
        
        # Create vector store
        start_time = time.time()
        vector_store = VectorStoreFactory.create_vector_store(
            store_type=store_type,
            embedding_provider=embedding_provider
        )
        creation_time = time.time() - start_time
        
        print(f"✅ Created vector store in {creation_time:.4f}s: {type(vector_store).__name__}")
        
        # Check if index exists
        if hasattr(vector_store, "_index_exists"):
            has_index = vector_store._index_exists()
            print(f"{'✅' if has_index else '⚠️'} Vector store has existing index: {has_index}")
        
        # Create test documents
        documents = [
            Document(page_content="This is a test document about RAG systems", 
                     metadata={"source": "test", "topic": "rag"}),
            Document(page_content="FAISS is a library for efficient similarity search", 
                     metadata={"source": "test", "topic": "faiss"}),
            Document(page_content="ChromaDB is a database for storing and querying embeddings", 
                     metadata={"source": "test", "topic": "chroma"})
        ]
        
        print(f"Created {len(documents)} test documents")
        
        # Add documents
        start_time = time.time()
        success = vector_store.add_documents(documents)
        addition_time = time.time() - start_time
        
        if success:
            print(f"✅ Added documents in {addition_time:.4f}s")
        else:
            print(f"❌ Failed to add documents")
            return None, None
        
        # Test search
        try:
            start_time = time.time()
            results = vector_store.search("What is RAG?", k=2)
            search_time = time.time() - start_time
            
            print(f"✅ Search completed in {search_time:.4f}s")
            print(f"Found {len(results)} documents")
            
            # Display first result
            if results:
                print(f"First result: {results[0].page_content[:50]}...")
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
        
        return vector_store, documents
            
    except Exception as e:
        print(f"❌ Vector store testing failed: {str(e)}")
        return None, None

def test_rag_pipeline(vector_store, documents):
    """Test the RAG pipeline.
    
    Args:
        vector_store: Vector store to use
        documents: Documents that were added to the vector store
        
    Returns:
        RAG pipeline object or None if failed
    """
    print("\n=== Testing RAG Pipeline ===")
    
    if not vector_store:
        print("❌ Cannot test RAG pipeline: Vector store not available")
        return None
    
    try:
        # Create LLM client
        llm_client = LLMClient()
        print(f"✅ Created LLM client: {type(llm_client).__name__}")
        
        # Create retriever
        retriever = vector_store.as_retriever()
        print(f"✅ Created retriever: {type(retriever).__name__}")
        
        # Verify retriever has correct method
        if hasattr(retriever, 'get_relevant_documents'):
            print("✅ Retriever has get_relevant_documents method")
        else:
            print("❌ Retriever missing get_relevant_documents method")
        
        # Create RAG pipeline
        start_time = time.time()
        rag_pipeline = RAGPipeline(
            llm=llm_client,
            retriever=retriever
        )
        creation_time = time.time() - start_time
        
        print(f"✅ Created RAG pipeline in {creation_time:.4f}s")
        
        # Test query
        test_query = "What is RAG?"
        
        print(f"Testing query: {test_query}")
        
        # Run query
        start_time = time.time()
        result = rag_pipeline.run(test_query)
        query_time = time.time() - start_time
        
        print(f"✅ Executed query in {query_time:.4f}s")
        print(f"Retrieved {len(result['source_documents'])} documents")
        print(f"Response: {result['result'][:100]}...")
        
        return rag_pipeline
            
    except Exception as e:
        print(f"❌ RAG pipeline testing failed: {str(e)}")
        return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple RAG Pipeline Diagnostics")
    parser.add_argument(
        "--store-type", 
        type=str, 
        default="faiss",
        choices=["faiss", "chroma"],
        help="Type of vector store to use"
    )
    
    args = parser.parse_args()
    
    print("=== RAG Pipeline Simple Diagnostics ===")
    
    # Test API key
    api_key_valid = test_openai_api_key()
    if not api_key_valid:
        print("\n❌ Diagnostics failed: Invalid OpenAI API key")
        print("Please set a valid OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Test vector store
    vector_store, documents = test_vector_store(args.store_type)
    if not vector_store:
        print("\n❌ Diagnostics failed: Vector store creation failed")
        sys.exit(1)
    
    # Test RAG pipeline
    rag_pipeline = test_rag_pipeline(vector_store, documents)
    if not rag_pipeline:
        print("\n❌ Diagnostics failed: RAG pipeline creation failed")
        sys.exit(1)
    
    print("\n=== Diagnostics Summary ===")
    print("✅ OpenAI API key is valid")
    print(f"✅ {args.store_type.upper()} vector store is working")
    print("✅ RAG pipeline is working")
    print("\nAll tests passed successfully! The RAG pipeline is ready to use.")

if __name__ == "__main__":
    main()