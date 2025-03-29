# core/schemas/content_schema.py

"""Content schema definitions for educational content."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict

class ContentType(Enum):
    """Types of educational content."""
    IEP = "iep"
    LESSON_PLAN = "lesson_plan"
    ASSESSMENT = "assessment"
    WORKSHEET = "worksheet"
    ACTIVITY = "activity"
    RESOURCE = "resource"
    NOTE = "note"
    OTHER = "other"

class DifficultyLevel(Enum):
    """Content difficulty levels."""
    INTRODUCTORY = "introductory"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class TimeFrame(Enum):
    """Lesson plan timeframes."""
    DAILY = "daily"
    WEEKLY = "weekly"
    UNIT = "unit"
    SEMESTER = "semester"
    YEAR = "year"

class AccessibilityFeature(Enum):
    """Content accessibility features."""
    VISUAL_SUPPORTS = "visual_supports"
    AUDIO_SUPPORTS = "audio_supports"
    SIMPLIFIED_LANGUAGE = "simplified_language"
    EXTENDED_TIME = "extended_time"
    CHUNKED_CONTENT = "chunked_content"
    MULTIPLE_REPRESENTATIONS = "multiple_representations"
    ASSISTIVE_TECHNOLOGY = "assistive_technology"
    OTHER = "other"

@dataclass
class ContentStandard:
    """Educational standard reference."""
    standard_id: str
    code: str
    description: str
    framework: str = ""  # e.g., "Common Core", "NGSS"
    subject: str = ""
    grade_level: str = ""

@dataclass
class ContentTag:
    """Content classification tag."""
    name: str
    category: str = ""  # e.g., "subject", "skill", "topic"
    value: str = ""  # Optional additional data

@dataclass
class ContentSection:
    """Section of educational content."""
    section_id: str
    title: str
    content: str
    order: int = 0
    content_type: str = "text"  # text, image_description, code, table, list
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentMetadata:
    """Metadata for educational content."""
    subject: str
    grade_level: str
    content_type: ContentType
    author_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    standards: List[ContentStandard] = field(default_factory=list)
    tags: List[ContentTag] = field(default_factory=list)
    difficulty: Optional[DifficultyLevel] = None
    estimated_duration: Optional[str] = None  # e.g., "45 minutes"
    keywords: List[str] = field(default_factory=list)
    related_kc_ids: List[str] = field(default_factory=list)  # Knowledge components
    accessibility_features: List[AccessibilityFeature] = field(default_factory=list)

@dataclass
class EducationalContent:
    """Base class for all educational content."""
    content_id: str
    title: str
    description: str
    metadata: ContentMetadata
    sections: List[ContentSection] = field(default_factory=list)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EducationalContent':
        """Create from dictionary.
        
        Args:
            data: Dictionary with content data
            
        Returns:
            EducationalContent instance
        """
        # Handle nested dataclasses and enums
        if 'metadata' in data:
            metadata_data = data['metadata']
            
            if 'content_type' in metadata_data:
                metadata_data['content_type'] = ContentType(metadata_data['content_type'])
            
            if 'difficulty' in metadata_data and metadata_data['difficulty'] is not None:
                metadata_data['difficulty'] = DifficultyLevel(metadata_data['difficulty'])
            
            if 'standards' in metadata_data:
                standards = []
                for standard_data in metadata_data['standards']:
                    standards.append(ContentStandard(**standard_data))
                metadata_data['standards'] = standards
            
            if 'tags' in metadata_data:
                tags = []
                for tag_data in metadata_data['tags']:
                    tags.append(ContentTag(**tag_data))
                metadata_data['tags'] = tags
            
            if 'accessibility_features' in metadata_data:
                features = []
                for feature in metadata_data['accessibility_features']:
                    features.append(AccessibilityFeature(feature))
                metadata_data['accessibility_features'] = features
            
            data['metadata'] = ContentMetadata(**metadata_data)
        
        if 'sections' in data:
            sections = []
            for section_data in data['sections']:
                sections.append(ContentSection(**section_data))
            data['sections'] = sections
        
        return cls(**data)

@dataclass
class IEPContent(EducationalContent):
    """Individualized Education Program content."""
    student_id: str = ""
    goals: List[Dict[str, Any]] = field(default_factory=list)
    accommodations: List[Dict[str, Any]] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_methods: List[Dict[str, Any]] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    review_date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IEPContent':
        """Create from dictionary with proper typing."""
        # First convert basic content fields
        base_content = EducationalContent.from_dict(data)
        
        # Extract IEP-specific fields
        iep_data = {
            'content_id': base_content.content_id,
            'title': base_content.title,
            'description': base_content.description,
            'metadata': base_content.metadata,
            'sections': base_content.sections,
            'version': base_content.version,
            'student_id': data.get('student_id', ''),
            'goals': data.get('goals', []),
            'accommodations': data.get('accommodations', []),
            'services': data.get('services', []),
            'evaluation_methods': data.get('evaluation_methods', []),
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'review_date': data.get('review_date')
        }
        
        return cls(**iep_data)

@dataclass
class LessonPlanContent(EducationalContent):
    """Lesson plan content."""
    timeframe: TimeFrame = TimeFrame.DAILY
    objectives: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    instructional_strategies: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    assessment_methods: List[Dict[str, Any]] = field(default_factory=list)
    differentiation: Dict[str, Any] = field(default_factory=dict)
    accommodations: List[Dict[str, Any]] = field(default_factory=list)
    schedule: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LessonPlanContent':
        """Create from dictionary with proper typing."""
        # First convert basic content fields
        base_content = EducationalContent.from_dict(data)
        
        # Handle timeframe enum
        timeframe = TimeFrame(data.get('timeframe', 'daily'))
        
        # Extract lesson plan specific fields
        lesson_data = {
            'content_id': base_content.content_id,
            'title': base_content.title,
            'description': base_content.description,
            'metadata': base_content.metadata,
            'sections': base_content.sections,
            'version': base_content.version,
            'timeframe': timeframe,
            'objectives': data.get('objectives', []),
            'materials': data.get('materials', []),
            'instructional_strategies': data.get('instructional_strategies', []),
            'activities': data.get('activities', []),
            'assessment_methods': data.get('assessment_methods', []),
            'differentiation': data.get('differentiation', {}),
            'accommodations': data.get('accommodations', []),
            'schedule': data.get('schedule', {})
        }
        
        return cls(**lesson_data)

@dataclass
class ContentRepository:
    """Repository of educational content."""
    contents: Dict[str, EducationalContent] = field(default_factory=dict)
    
    def add_content(self, content: EducationalContent) -> None:
        """Add content to the repository.
        
        Args:
            content: The content to add
        """
        self.contents[content.content_id] = content
    
    def get_content(self, content_id: str) -> Optional[EducationalContent]:
        """Get content by ID.
        
        Args:
            content_id: Content ID
            
        Returns:
            Content or None if not found
        """
        return self.contents.get(content_id)
    
    def get_contents_by_type(self, content_type: ContentType) -> List[EducationalContent]:
        """Get contents by type.
        
        Args:
            content_type: Content type
            
        Returns:
            List of matching contents
        """
        return [
            content for content in self.contents.values()
            if content.metadata.content_type == content_type
        ]
    
    def get_contents_by_student(self, student_id: str) -> List[EducationalContent]:
        """Get contents for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            List of matching contents
        """
        return [
            content for content in self.contents.values()
            if isinstance(content, IEPContent) and content.student_id == student_id
        ]
    
    def search_contents(self, query: Dict[str, Any]) -> List[EducationalContent]:
        """Search contents by metadata fields.
        
        Args:
            query: Dictionary of field-value pairs to match
            
        Returns:
            List of matching contents
        """
        results = []
        
        for content in self.contents.values():
            match = True
            
            for field, value in query.items():
                # Handle nested fields with dot notation
                if '.' in field:
                    parts = field.split('.')
                    obj = content
                    
                    for part in parts:
                        if hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            match = False
                            break
                    
                    if match and obj != value:
                        match = False
                
                # Handle direct fields
                elif hasattr(content, field):
                    if getattr(content, field) != value:
                        match = False
                
                # Handle metadata fields
                elif hasattr(content.metadata, field):
                    if getattr(content.metadata, field) != value:
                        match = False
                
                else:
                    match = False
            
            if match:
                results.append(content)
        
        return results