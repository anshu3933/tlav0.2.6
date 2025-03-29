# test_add_documents.py
from langchain.schema import Document
from core.embeddings.vector_store import FAISSVectorStore
import os
import shutil

# Temporarily rename any existing index directory for testing
if os.path.exists("models/faiss_index"):
    if os.path.exists("models/faiss_index_backup"):
        shutil.rmtree("models/faiss_index_backup")
    os.rename("models/faiss_index", "models/faiss_index_backup")
    
# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Create a new vector store (without Fix 1)
vector_store = FAISSVectorStore()

# Create test documents
test_docs = [
    Document(page_content="This is a test document for vector store", metadata={"source": "test1.txt"}),
    Document(page_content="Another test document for FAISS indexing", metadata={"source": "test2.txt"})
]

# Add documents (should build a new index)
success = vector_store.add_documents(test_docs)

# Check results
if success and vector_store._index_exists():
    print("✅ Success: Documents added and index created successfully")
else:
    print("❌ Error: Failed to add documents or create index")

# Restore original index directory if it existed
if os.path.exists("models/faiss_index_backup"):
    if os.path.exists("models/faiss_index"):
        shutil.rmtree("models/faiss_index")
    os.rename("models/faiss_index_backup", "models/faiss_index")