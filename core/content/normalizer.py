# core/content/normalizer.py

"""Content normalizer for standardizing educational content."""

import re
from typing import Dict, List, Any, Optional, TypeVar, Union
from config.logging_config import get_module_logger
from core.schemas.content_schema import (
    EducationalContent, IEPContent, LessonPlanContent,
    ContentSection, ContentType
)
from core.content.strategies import (
    NormalizationStrategy, DefaultNormalizationStrategy,
    IEPNormalizationStrategy, LessonPlanNormalizationStrategy
)

# Create a logger for this module
logger = get_module_logger("content_normalizer")

# Type variable for content types
T = TypeVar('T', bound=EducationalContent)

class ContentNormalizer:
    """Normalizes educational content using configurable strategies."""
    
    def __init__(self):
        """Initialize the normalizer with strategies."""
        self._strategies: Dict[ContentType, NormalizationStrategy] = {
            ContentType.IEP: IEPNormalizationStrategy(),
            ContentType.LESSON_PLAN: LessonPlanNormalizationStrategy(),
            # Default strategy for other content types
            None: DefaultNormalizationStrategy()
        }
        
        logger.debug("Initialized content normalizer")
    
    def register_strategy(self, content_type: ContentType, strategy: NormalizationStrategy) -> None:
        """Register a normalization strategy for a content type.
        
        Args:
            content_type: Content type
            strategy: Normalization strategy
        """
        self._strategies[content_type] = strategy
        logger.debug(f"Registered normalization strategy for {content_type.name}")
    
    def normalize(self, content: T) -> T:
        """Normalize content using the appropriate strategy.
        
        Args:
            content: Content to normalize
            
        Returns:
            Normalized content
        """
        try:
            # Determine content type
            content_type = content.metadata.content_type
            
            # Get appropriate strategy
            strategy = self._strategies.get(content_type, self._strategies[None])
            
            # Apply strategy
            normalized_content = strategy.normalize(content)
            
            # Apply common normalizations
            normalized_content = self._apply_common_normalizations(normalized_content)
            
            logger.debug(f"Normalized {content_type.name} content: {content.content_id}")
            return normalized_content
            
        except Exception as e:
            logger.error(f"Error normalizing content: {str(e)}", exc_info=True)
            # Return original content on error
            return content
    
    def _apply_common_normalizations(self, content: T) -> T:
        """Apply common normalizations to any content type.
        
        Args:
            content: Content to normalize
            
        Returns:
            Normalized content
        """
        # Title normalization (capitalize first letter of each word)
        content.title = self._normalize_title(content.title)
        
        # Section content normalization
        for section in content.sections:
            section.content = self._normalize_section_content(section)
        
        return content
    
    def _normalize_title(self, title: str) -> str:
        """Normalize a title.
        
        Args:
            title: Title to normalize
            
        Returns:
            Normalized title
        """
        # Simple title case with some exceptions
        exceptions = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 
                      'to', 'from', 'by', 'in', 'of', 'with', 'as'}
        
        words = title.split()
        
        # Always capitalize first and last words
        if words:
            words[0] = words[0].capitalize()
            
        if len(words) > 1:
            words[-1] = words[-1].capitalize()
        
        # Handle other words
        for i in range(1, len(words) - 1):
            word = words[i]
            if word.lower() not in exceptions:
                words[i] = word.capitalize()
            else:
                words[i] = word.lower()
        
        return ' '.join(words)
    
    def _normalize_section_content(self, section: ContentSection) -> str:
        """Normalize section content based on content type.
        
        Args:
            section: Section to normalize
            
        Returns:
            Normalized content
        """
        content = section.content
        content_type = section.content_type
        
        if content_type == "text":
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Ensure proper sentence capitalization
            sentences = re.split(r'(?<=[.!?])\s+', content)
            sentences = [s.capitalize() for s in sentences if s]
            content = ' '.join(sentences)
        
        elif content_type == "list":
            # Ensure consistent list formatting
            items = [line.strip() for line in content.split('\n') if line.strip()]
            items = [f"- {item}" if not item.startswith(('-', '*', '•')) else item for item in items]
            content = '\n'.join(items)
        
        return content

# Create a singleton instance
content_normalizer = ContentNormalizer()