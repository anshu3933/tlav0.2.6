# config/vertex_config.py

import os
from dataclasses import dataclass

@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI."""
    project_id: str = os.environ.get("GCP_PROJECT", "your-project-id")
    location: str = os.environ.get("GCP_LOCATION", "us-central1")
    index_name: str = os.environ.get("VERTEX_INDEX_NAME", "educational-assistant-index")
    embedding_model: str = os.environ.get("VERTEX_EMBEDDING_MODEL", "textembedding-gecko@latest")
    
    # API request parameters
    request_timeout: int = 60
    max_retries: int = 3
    
    # Cost control
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour

# Create a singleton instance
vertex_config = VertexAIConfig()