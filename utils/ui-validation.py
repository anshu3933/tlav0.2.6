"""Utilities for UI component validation and error handling."""

from typing import Dict, Any, Optional, List, Tuple, Union
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("ui_validation")

def validate_response_structure(response: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate the structure of a response from a RAG chain.
    
    Args:
        response: Response to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(response, dict):
        return False, f"Invalid response format: expected dictionary, got {type(response)}"
    
    if "result" not in response:
        return False, "Response missing required 'result' field"
    
    return True, None

def validate_document_structure(doc: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the structure of a document and extract relevant fields.
    
    Args:
        doc: Document to validate
        
    Returns:
        Tuple of (is_valid, extracted_fields)
    """
    result = {"is_valid": True, "content": None, "metadata": {}}
    
    # Check document content
    if hasattr(doc, 'page_content'):
        result["content"] = doc.page_content
    else:
        result["is_valid"] = False
        result["error"] = "Document missing page_content field"
    
    # Check document metadata
    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
        result["metadata"] = doc.metadata
    
    return result["is_valid"], result
