# core/schemas/student_schema.py

"""Student schema definitions for data validation and conversion."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict

class LearningStyle(Enum):
    """Learning style classification."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"

class AccommodationType(Enum):
    """Types of educational accommodations."""
    PRESENTATION = "presentation"  # How information is presented to the student
    RESPONSE = "response"  # How the student demonstrates learning
    SETTING = "setting"  # Where the student learns and is assessed
    TIMING = "timing"  # When the student learns and is assessed
    ORGANIZATIONAL = "organizational"  # Help with organization and planning
    BEHAVIORAL = "behavioral"  # Support for appropriate behavior
    ASSISTIVE = "assistive"  # Assistive technology or devices
    OTHER = "other"  # Other accommodations

@dataclass
class Accommodation:
    """Educational accommodation details."""
    accommodation_id: str
    type: AccommodationType
    description: str
    instructions: str
    subject_specific: bool = False
    subjects: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    active: bool = True

@dataclass
class KnowledgeComponent:
    """Knowledge component for knowledge tracing."""
    kc_id: str
    name: str
    description: str
    prerequisites: List[str] = field(default_factory=list)
    category: str = ""
    subcategory: str = ""
    grade_level: Optional[str] = None
    difficulty: float = 0.5  # 0.0 (easy) to 1.0 (difficult)

@dataclass
class KnowledgeState:
    """Student's knowledge state for a specific knowledge component."""
    kc_id: str
    mastery: float = 0.0  # 0.0 (not mastered) to 1.0 (fully mastered)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0  # Confidence in the mastery estimate
    # BKT parameters
    prior: float = 0.0  # Prior probability of mastery
    learn: float = 0.1  # Probability of learning
    slip: float = 0.1  # Probability of slipping
    guess: float = 0.2  # Probability of guessing
    # Learning trajectory
    assessment_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class Goal:
    """Educational goal for a student."""
    goal_id: str
    description: str
    related_kc_ids: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    target_date: Optional[str] = None
    status: str = "active"  # active, completed, abandoned
    progress: float = 0.0  # 0.0 to 1.0
    assessment_criteria: str = ""

@dataclass
class Assessment:
    """Assessment record."""
    assessment_id: str
    type: str  # quiz, test, project, observation
    date: str
    score: Optional[float] = None
    max_score: Optional[float] = None
    kc_results: Dict[str, float] = field(default_factory=dict)  # KC ID -> score
    notes: str = ""

@dataclass
class StudentProfile:
    """Student profile for personalized learning."""
    student_id: str
    name: str
    grade_level: str
    # Demographic information
    birth_date: Optional[str] = None
    gender: Optional[str] = None
    # Learning profile
    learning_style: Optional[LearningStyle] = None
    strengths: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    # IEP information
    has_iep: bool = False
    iep_id: Optional[str] = None
    # Accommodations
    accommodations: List[Accommodation] = field(default_factory=list)
    # Knowledge state
    knowledge_states: Dict[str, KnowledgeState] = field(default_factory=dict)
    # Goals
    goals: List[Goal] = field(default_factory=list)
    # Assessment history
    assessments: List[Assessment] = field(default_factory=list)
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentProfile':
        """Create from dictionary.
        
        Args:
            data: Dictionary with student profile data
            
        Returns:
            StudentProfile instance
        """
        # Handle nested dataclasses and enums
        if 'learning_style' in data and data['learning_style'] is not None:
            data['learning_style'] = LearningStyle(data['learning_style'])
        
        if 'accommodations' in data:
            accommodations = []
            for acc_data in data['accommodations']:
                if 'type' in acc_data:
                    acc_data['type'] = AccommodationType(acc_data['type'])
                accommodations.append(Accommodation(**acc_data))
            data['accommodations'] = accommodations
        
        if 'knowledge_states' in data:
            knowledge_states = {}
            for kc_id, ks_data in data['knowledge_states'].items():
                knowledge_states[kc_id] = KnowledgeState(**ks_data)
            data['knowledge_states'] = knowledge_states
        
        if 'goals' in data:
            goals = []
            for goal_data in data['goals']:
                goals.append(Goal(**goal_data))
            data['goals'] = goals
        
        if 'assessments' in data:
            assessments = []
            for assessment_data in data['assessments']:
                assessments.append(Assessment(**assessment_data))
            data['assessments'] = assessments
        
        return cls(**data)
    
    def update_knowledge_state(self, kc_id: str, assessment_result: float) -> None:
        """Update knowledge state based on assessment result.
        
        Args:
            kc_id: Knowledge component ID
            assessment_result: Assessment result (0.0 to 1.0)
        """
        # Create knowledge state if it doesn't exist
        if kc_id not in self.knowledge_states:
            self.knowledge_states[kc_id] = KnowledgeState(kc_id=kc_id)
        
        # Get current knowledge state
        ks = self.knowledge_states[kc_id]
        
        # Add assessment to history
        ks.assessment_history.append({
            "timestamp": datetime.now().isoformat(),
            "result": assessment_result
        })
        
        # Update mastery using simple BKT update
        if assessment_result >= 0.7:  # Correct response threshold
            # Update mastery using the learning rate
            ks.mastery = ks.mastery + (1 - ks.mastery) * ks.learn
        else:
            # Adjust for slipping
            ks.mastery = ks.mastery * (1 - ks.slip) / (ks.mastery * (1 - ks.slip) + (1 - ks.mastery) * ks.guess)
        
        # Update last updated timestamp
        ks.last_updated = datetime.now().isoformat()
        
        # Update profile updated timestamp
        self.updated_at = datetime.now().isoformat()

@dataclass
class ClassProfile:
    """Class profile for group analytics."""
    class_id: str
    name: str
    grade_level: str
    subject: Optional[str] = None
    teacher_id: Optional[str] = None
    student_ids: List[str] = field(default_factory=list)
    # Class-level knowledge components and analytics
    target_kcs: List[str] = field(default_factory=list)
    class_analytics: Dict[str, Any] = field(default_factory=dict)
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassProfile':
        """Create from dictionary.
        
        Args:
            data: Dictionary with class profile data
            
        Returns:
            ClassProfile instance
        """
        return cls(**data)