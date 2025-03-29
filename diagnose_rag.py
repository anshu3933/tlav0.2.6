# diagnose_rag.py

"""Diagnostic script for testing and debugging the RAG pipeline."""

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
from fix_vector_store import verify_store_type, create_empty_faiss_index

# Create a logger for this module
logger = get_module_logger("diagnose_rag")

class RAGDiagnostics:
    """Diagnostic tools for the RAG pipeline."""
    
    def __init__(self, 
                 store_type: str = "faiss",
                 verbose: bool = False):
        """Initialize with parameters.
        
        Args:
            store_type: Type of vector store to use
            verbose: Whether to print verbose output
        """
        self.store_type = store_type
        self.verbose = verbose
        self.results = {}
        
        # Configure logging
        if verbose:
            print(f"Initializing RAG diagnostics with store type: {store_type}")
    
    def run_diagnostics(self):
        """Run a complete diagnostic test."""
        self.print_step("Starting RAG diagnostics")
        
        # Test vector store creation
        vector_store = self.test_vector_store_creation()
        
        # Test document addition
        if vector_store:
            self.test_document_addition(vector_store)
        
        # Test RAG pipeline
        if vector_store:
            self.test_rag_pipeline(vector_store)
        
        # Print summary
        self.print_summary()
    
    def test_vector_store_creation(self):
        """Test vector store creation.
        
        Returns:
            Vector store instance or None if failed
        """
        self.print_step("Testing vector store creation")
        
        try:
            # Verify store type
            if not verify_store_type(self.store_type):
                print(f"⚠️ Vector store type {self.store_type} verification failed")
                
                # Try to initialize FAISS index as fallback
                if self.store_type == "faiss":
                    print("Attempting to initialize empty FAISS index...")
                    create_empty_faiss_index()
            
            # Create embedding provider
            embedding_provider = VectorStoreFactory.create_embeddings_provider()
            print(f"✅ Created embedding provider: {type(embedding_provider).__name__}")
            
            # Create vector store
            start_time = time.time()
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=self.store_type,
                embedding_provider=embedding_provider
            )
            creation_time = time.time() - start_time
            
            print(f"✅ Created vector store: {type(vector_store).__name__} in {creation_time:.4f}s")
            
            # Check if index exists
            if hasattr(vector_store, "_index_exists"):
                has_index = vector_store._index_exists()
                print(f"{'✅' if has_index else '⚠️'} Vector store has existing index: {has_index}")
            
            # Record results
            self.results["vector_store_creation"] = {
                "success": True,
                "type": type(vector_store).__name__,
                "creation_time": creation_time,
                "has_index": has_index if "has_index" in locals() else None
            }
            
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            
            # Record results
            self.results["vector_store_creation"] = {
                "success": False,
                "error": str(e)
            }
            
            return None
    
    def test_document_addition(self, vector_store):
        """Test adding documents to the vector store.
        
        Args:
            vector_store: Vector store instance
        """
        self.print_step("Testing document addition")
        
        try:
            # Create test documents
            documents = [
                Document(page_content="This is a test document about RAG systems", metadata={"source": "test", "topic": "rag"}),
                Document(page_content="FAISS is a library for efficient similarity search", metadata={"source": "test", "topic": "faiss"}),
                Document(page_content="ChromaDB is a database for storing and querying embeddings", metadata={"source": "test", "topic": "chroma"})
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
            
            # Record results
            self.results["document_addition"] = {
                "success": success,
                "document_count": len(documents),
                "addition_time": addition_time
            }
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            
            # Record results
            self.results["document_addition"] = {
                "success": False,
                "error": str(e)
            }
    
    def test_rag_pipeline(self, vector_store):
        """Test the RAG pipeline.
        
        Args:
            vector_store: Vector store instance
        """
        self.print_step("Testing RAG pipeline")
        
        try:
            # Create LLM client
            llm_client = LLMClient()
            print(f"✅ Created LLM client")
            
            # Create observability
            rag_observability = RagObservability()
            print(f"✅ Created RAG observability")
            
            # Create RAG pipeline
            start_time = time.time()
            rag_pipeline = RAGPipeline(
                llm=llm_client,
                retriever=vector_store.as_retriever(),
                observability_callbacks=[rag_observability.rag_step_callback()]
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
            
            # Record results
            self.results["rag_pipeline"] = {
                "success": True,
                "creation_time": creation_time,
                "query_time": query_time,
                "document_count": len(result['source_documents']),
                "response_sample": result['result'][:100]
            }
            
        except Exception as e:
            print(f"❌ Error testing RAG pipeline: {str(e)}")
            
            # Record results
            self.results["rag_pipeline"] = {
                "success": False,
                "error": str(e)
            }
    
    def print_step(self, message: str):
        """Print a step message."""
        print(f"\n=== {message} ===")
    
    def print_summary(self):
        """Print a summary of the diagnostic results."""
        self.print_step("Diagnostic Summary")
        
        # Check results
        vector_store_success = self.results.get("vector_store_creation", {}).get("success", False)
        document_success = self.results.get("document_addition", {}).get("success", False)
        pipeline_success = self.results.get("rag_pipeline", {}).get("success", False)
        
        # Print overall success
        if vector_store_success and document_success and pipeline_success:
            print("🎉 All diagnostics passed successfully!")
        else:
            print("⚠️ Some diagnostics failed:")
            if not vector_store_success:
                print("  ❌ Vector store creation failed")
            if not document_success:
                print("  ❌ Document addition failed")
            if not pipeline_success:
                print("  ❌ RAG pipeline test failed")
        
        # Save results to file
        self.save_results()
    
    def save_results(self, filename: str = "rag_diagnostics.json"):
        """Save diagnostic results to a JSON file.
        
        Args:
            filename: Name of the file to save results to
        """
        # Add timestamp
        self.results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to file
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Diagnostic results saved to {filename}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Pipeline Diagnostics")
    parser.add_argument(
        "--store-type", 
        type=str, 
        default="faiss",
        choices=["faiss", "chroma", "vertex"],
        help="Type of vector store to use"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = RAGDiagnostics(
        store_type=args.store_type,
        verbose=args.verbose
    )
    diagnostics.run_diagnostics()

if __name__ == "__main__":
    main()            # Recor