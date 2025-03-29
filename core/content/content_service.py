# core/content/content_service.py

"""Content service for managing educational content."""

import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from config.logging_config import get_module_logger
from core.schemas.content_schema import (
    EducationalContent, IEPContent, LessonPlanContent, ContentType,
    ContentMetadata, ContentSection, ContentStandard, ContentTag,
    DifficultyLevel, TimeFrame, AccessibilityFeature
)
from core.content.normalizer import ContentNormalizer
from core.events.typed_event_bus import Event, event_bus

# Create a logger for this module
logger = get_module_logger("content_service")

@Event
class ContentCreatedEvent:
    """Event fired when content is created."""
    content_id: str
    content_type: str
    title: str
    created_by: str
    timestamp: str = datetime.now().isoformat()

@Event
class ContentUpdatedEvent:
    """Event fired when content is updated."""
    content_id: str
    content_type: str
    title: str
    updated_by: str
    changes: Dict[str, Any]
    timestamp: str = datetime.now().isoformat()

@Event
class ContentDeletedEvent:
    """Event fired when content is deleted."""
    content_id: str
    content_type: str
    deleted_by: str
    timestamp: str = datetime.now().isoformat()

class ContentService:
    """Service for managing educational content."""
    
    def __init__(self, content_normalizer: Optional[ContentNormalizer] = None):
        """Initialize the content service.
        
        Args:
            content_normalizer: Optional content normalizer for standardizing content
        """
        self.content_normalizer = content_normalizer
        self._contents: Dict[str, EducationalContent] = {}
        logger.debug("Initialized content service")
    
    def create_iep(self, 
                  title: str,
                  description: str,
                  student_id: str,
                  subject: str,
                  grade_level: str,
                  goals: List[Dict[str, Any]],
                  accommodations: List[Dict[str, Any]],
                  services: List[Dict[str, Any]] = None,
                  evaluation_methods: List[Dict[str, Any]] = None,
                  sections: List[Dict[str, Any]] = None,
                  author_id: str = None,
                  **kwargs) -> IEPContent:
        """Create an IEP content.
        
        Args:
            title: Content title
            description: Content description
            student_id: Student ID
            subject: Subject area
            grade_level: Grade level
            goals: List of goals
            accommodations: List of accommodations
            services: Optional list of services
            evaluation_methods: Optional list of evaluation methods
            sections: Optional content sections
            author_id: Optional author ID
            **kwargs: Additional fields
            
        Returns:
            Created IEP content
        """
        # Generate content ID
        content_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = ContentMetadata(
            subject=subject,
            grade_level=grade_level,
            content_type=ContentType.IEP,
            author_id=author_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Process sections
        content_sections = []
        if sections:
            for i, section_data in enumerate(sections):
                section_id = section_data.get('section_id', str(uuid.uuid4()))
                section = ContentSection(
                    section_id=section_id,
                    title=section_data.get('title', f"Section {i+1}"),
                    content=section_data.get('content', ""),
                    order=section_data.get('order', i),
                    content_type=section_data.get('content_type', "text"),
                    metadata=section_data.get('metadata', {})
                )
                content_sections.append(section)
        
        # Create IEP content
        iep = IEPContent(
            content_id=content_id,
            title=title,
            description=description,
            metadata=metadata,
            sections=content_sections,
            student_id=student_id,
            goals=goals or [],
            accommodations=accommodations or [],
            services=services or [],
            evaluation_methods=evaluation_methods or [],
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date'),
            review_date=kwargs.get('review_date')
        )
        
        # Normalize content if normalizer is available
        if self.content_normalizer:
            iep = self.content_normalizer.normalize(iep)
        
        # Store content
        self._contents[content_id] = iep
        
        # Publish event
        event_bus.publish(ContentCreatedEvent(
            content_id=content_id,
            content_type="iep",
            title=title,
            created_by=author_id or "system"
        ))
        
        logger.info(f"Created IEP content: {content_id} - {title}")
        return iep
    
    def create_lesson_plan(self,
                          title: str,
                          description: str,
                          subject: str,
                          grade_level: str,
                          timeframe: Union[str, TimeFrame],
                          objectives: List[str],
                          materials: List[str] = None,
                          instructional_strategies: List[Dict[str, Any]] = None,
                          activities: List[Dict[str, Any]] = None,
                          assessment_methods: List[Dict[str, Any]] = None,
                          differentiation: Dict[str, Any] = None,
                          accommodations: List[Dict[str, Any]] = None,
                          schedule: Dict[str, Any] = None,
                          sections: List[Dict[str, Any]] = None,
                          author_id: str = None,
                          **kwargs) -> LessonPlanContent:
        """Create a lesson plan content.
        
        Args:
            title: Content title
            description: Content description
            subject: Subject area
            grade_level: Grade level
            timeframe: Lesson plan timeframe
            objectives: List of learning objectives
            materials: Optional list of materials
            instructional_strategies: Optional instructional strategies
            activities: Optional list of activities
            assessment_methods: Optional assessment methods
            differentiation: Optional differentiation strategies
            accommodations: Optional accommodations
            schedule: Optional schedule
            sections: Optional content sections
            author_id: Optional author ID
            **kwargs: Additional fields
            
        Returns:
            Created lesson plan content
        """
        # Generate content ID
        content_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = ContentMetadata(
            subject=subject,
            grade_level=grade_level,
            content_type=ContentType.LESSON_PLAN,
            author_id=author_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            estimated_duration=kwargs.get('duration')
        )
        
        # Process sections
        content_sections = []
        if sections:
            for i, section_data in enumerate(sections):
                section_id = section_data.get('section_id', str(uuid.uuid4()))
                section = ContentSection(
                    section_id=section_id,
                    title=section_data.get('title', f"Section {i+1}"),
                    content=section_data.get('content', ""),
                    order=section_data.get('order', i),
                    content_type=section_data.get('content_type', "text"),
                    metadata=section_data.get('metadata', {})
                )
                content_sections.append(section)
        
        # Handle timeframe enum
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe.lower())
        
        # Create lesson plan content
        lesson_plan = LessonPlanContent(
            content_id=content_id,
            title=title,
            description=description,
            metadata=metadata,
            sections=content_sections,
            timeframe=timeframe,
            objectives=objectives or [],
            materials=materials or [],
            instructional_strategies=instructional_strategies or [],
            activities=activities or [],
            assessment_methods=assessment_methods or [],
            differentiation=differentiation or {},
            accommodations=accommodations or [],
            schedule=schedule or {}
        )
        
        # Normalize content if normalizer is available
        if self.content_normalizer:
            lesson_plan = self.content_normalizer.normalize(lesson_plan)
        
        # Store content
        self._contents[content_id] = lesson_plan
        
        # Publish event
        event_bus.publish(ContentCreatedEvent(
            content_id=content_id,
            content_type="lesson_plan",
            title=title,
            created_by=author_id or "system"
        ))
        
        logger.info(f"Created lesson plan content: {content_id} - {title}")
        return lesson_plan
    
    def get_content(self, content_id: str) -> Optional[EducationalContent]:
        """Get content by ID.
        
        Args:
            content_id: Content ID
            
        Returns:
            Content or None if not found
        """
        return self._contents.get(content_id)
    
    def update_content(self, 
                      content_id: str, 
                      updates: Dict[str, Any], 
                      updated_by: str = "system") -> Optional[EducationalContent]:
        """Update content.
        
        Args:
            content_id: Content ID
            updates: Dictionary of updates
            updated_by: User ID who made the update
            
        Returns:
            Updated content or None if not found
        """
        content = self.get_content(content_id)
        if not content:
            logger.warning(f"Content not found for update: {content_id}")
            return None
        
        changes = {}
        
        # Handle metadata updates
        if 'metadata' in updates:
            metadata_updates = updates['metadata']
            for key, value in metadata_updates.items():
                if hasattr(content.metadata, key):
                    old_value = getattr(content.metadata, key)
                    setattr(content.metadata, key, value)
                    changes[f"metadata.{key}"] = {'old': old_value, 'new': value}
        
        # Handle direct field updates
        for key, value in updates.items():
            if key == 'metadata':
                continue
                
            if hasattr(content, key):
                old_value = getattr(content, key)
                setattr(content, key, value)
                changes[key] = {'old': old_value, 'new': value}
        
        # Update timestamp
        content.metadata.updated_at = datetime.now().isoformat()
        
        # Normalize content if normalizer is available
        if self.content_normalizer:
            content = self.content_normalizer.normalize(content)
        
        # Publish event
        event_bus.publish(ContentUpdatedEvent(
            content_id=content_id,
            content_type=content.metadata.content_type.value,
            title=content.title,
            updated_by=updated_by,
            changes=changes,
            timestamp=datetime.now().isoformat()
        ))
        
        logger.info(f"Updated content: {content_id} - {content.title}")
        return content
    
    def delete_content(self, content_id: str, deleted_by: str = "system") -> bool:
        """Delete content.
        
        Args:
            content_id: Content ID
            deleted_by: User ID who deleted the content
            
        Returns:
            True if deleted, False if not found
        """
        content = self.get_content(content_id)
        if not content:
            logger.warning(f"Content not found for deletion: {content_id}")
            return False
        
        # Remove content
        content_type = content.metadata.content_type.value
        title = content.title
        del self._contents[content_id]
        
        # Publish event
        event_bus.publish(ContentDeletedEvent(
            content_id=content_id,
            content_type=content_type,
            deleted_by=deleted_by,
            timestamp=datetime.now().isoformat()
        ))
        
        logger.info(f"Deleted content: {content_id} - {title}")
        return True
    
    def get_contents_by_type(self, content_type: Union[str, ContentType]) -> List[EducationalContent]:
        """Get contents by type.
        
        Args:
            content_type: Content type string or enum
            
        Returns:
            List of matching contents
        """
        # Convert string to enum if needed
        if isinstance(content_type, str):
            content_type = ContentType(content_type.lower())
        
        return [
            content for content in self._contents.values()
            if content.metadata.content_type == content_type
        ]
    
    def get_ieps_by_student(self, student_id: str) -> List[IEPContent]:
        """Get IEPs for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            List of matching IEPs
        """
        return [
            content for content in self._contents.values()
            if isinstance(content, IEPContent) and content.student_id == student_id
        ]
    
    def search_contents(self, query: Dict[str, Any]) -> List[EducationalContent]:
        """Search contents by criteria.
        
        Args:
            query: Dictionary of search criteria
            
        Returns:
            List of matching contents
        """
        results = []
        
        for content in self._contents.values():
            match = True
            
            for field, value in query.items():
                # Handle metadata fields
                if field.startswith('metadata.'):
                    metadata_field = field.split('.', 1)[1]
                    if not hasattr(content.metadata, metadata_field):
                        match = False
                        break
                    
                    field_value = getattr(content.metadata, metadata_field)
                    
                    # Handle enum values
                    if isinstance(field_value, Enum):
                        field_value = field_value.value
                    
                    if field_value != value:
                        match = False
                        break
                
                # Handle direct fields
                elif hasattr(content, field):
                    field_value = getattr(content, field)
                    
                    # Handle enum values
                    if isinstance(field_value, Enum):
                        field_value = field_value.value
                    
                    if field_value != value:
                        match = False
                        break
                
                # Field not found
                else:
                    match = False
                    break
            
            if match:
                results.append(content)
        
        return results

# Create a singleton instance
content_service = ContentService()