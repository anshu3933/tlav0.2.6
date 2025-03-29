# test_vector_store.py
from core.embeddings.vector_store import FAISSVectorStore

# Initialize vector store (should create an empty index)
vector_store = FAISSVectorStore()

# Check if index exists
if vector_store._index_exists():
    print("✅ Success: Empty index was created during initialization")
else:
    print("❌ Error: Failed to create empty index")