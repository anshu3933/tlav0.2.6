# core/llm/dspy/config.py

"""Configuration for DSPy framework integration."""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("dspy_config")

@dataclass
class DSPyPromptTemplate:
    """Template for a DSPy prompt."""
    name: str
    template: str
    description: str = ""
    version: str = "1.0"
    parameters: List[str] = field(default_factory=list)
    example_inputs: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSPyPromptTemplate':
        """Create from dictionary."""
        return cls(**data)
    
    def format(self, **kwargs) -> str:
        """Format the template with parameters.
        
        Args:
            **kwargs: Parameter values
            
        Returns:
            Formatted prompt
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing parameter in prompt template: {e}")
            # Return template with missing parameters marked
            return self.template.format(**{p: f"[MISSING {p}]" for p in self.parameters})
        except Exception as e:
            logger.error(f"Error formatting prompt template: {e}")
            return self.template

@dataclass
class DSPyConfig:
    """Configuration for DSPy framework."""
    default_model: str = "gpt-4o"
    prompt_templates: Dict[str, DSPyPromptTemplate] = field(default_factory=dict)
    max_tokens: int = 2000
    temperature: float = 0.7
    persistent_workspace: str = ".dspy_workspace"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_model": self.default_model,
            "prompt_templates": {name: template.to_dict() for name, template in self.prompt_templates.items()},
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "persistent_workspace": self.persistent_workspace
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSPyConfig':
        """Create from dictionary."""
        templates = {}
        if "prompt_templates" in data:
            for name, template_data in data["prompt_templates"].items():
                templates[name] = DSPyPromptTemplate.from_dict(template_data)
            
            # Remove templates to avoid double processing
            data_copy = data.copy()
            del data_copy["prompt_templates"]
            
            return cls(prompt_templates=templates, **data_copy)
        
        return cls(**data)
    
    def save(self, filepath: str) -> bool:
        """Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving DSPy configuration: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> Optional['DSPyConfig']:
        """Load configuration from file.
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            DSPyConfig or None if loading failed
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading DSPy configuration: {e}")
            return None
    
    def add_template(self, template: DSPyPromptTemplate) -> None:
        """Add a prompt template.
        
        Args:
            template: Prompt template to add
        """
        self.prompt_templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[DSPyPromptTemplate]:
        """Get a prompt template by name.
        
        Args:
            name: Template name
            
        Returns:
            Prompt template or None if not found
        """
        return self.prompt_templates.get(name)
    
    def remove_template(self, name: str) -> bool:
        """Remove a prompt template.
        
        Args:
            name: Template name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.prompt_templates:
            del self.prompt_templates[name]
            return True
        return False


# Create default configuration with educational templates
def create_default_config() -> DSPyConfig:
    """Create default DSPy configuration with educational templates."""
    config = DSPyConfig()
    
    # Add CoT template
    cot_template = DSPyPromptTemplate(
        name="educational_cot",
        description="Chain of thought template for educational reasoning",
        template="""
        Think step by step to solve this educational problem.
        
        Problem:
        {problem}
        
        Please show your reasoning in detail.
        """,
        parameters=["problem"],
        example_inputs={"problem": "How can we differentiate instruction for visual learners?"}
    )
    config.add_template(cot_template)
    
    # Add IEP analysis template
    iep_template = DSPyPromptTemplate(
        name="iep_analysis",
        description="Template for analyzing student IEPs",
        template="""
        Analyze the following IEP carefully to extract key information.
        
        IEP Content:
        {iep_content}
        
        Please extract the following information:
        1. Student strengths
        2. Areas of need
        3. Accommodations
        4. Goals
        5. Services
        """,
        parameters=["iep_content"],
        example_inputs={"iep_content": "Student X has an IEP that includes..."}
    )
    config.add_template(iep_template)
    
    # Add lesson planning template
    lesson_template = DSPyPromptTemplate(
        name="lesson_planning",
        description="Template for creating differentiated lesson plans",
        template="""
        Create a detailed lesson plan for the following specifications.
        
        Subject: {subject}
        Grade Level: {grade_level}
        Duration: {duration}
        Student Needs: {student_needs}
        
        Your lesson plan should include:
        1. Clear learning objectives
        2. Differentiated activities
        3. Assessment strategies
        4. Specific accommodations for student needs
        """,
        parameters=["subject", "grade_level", "duration", "student_needs"],
        example_inputs={
            "subject": "Mathematics",
            "grade_level": "5th Grade",
            "duration": "45 minutes",
            "student_needs": "Visual processing difficulties, ADHD"
        }
    )
    config.add_template(lesson_template)
    
    # Add knowledge tracing template
    kt_template = DSPyPromptTemplate(
        name="knowledge_tracing",
        description="Template for knowledge tracing analysis",
        template="""
        Analyze this student's performance history to determine mastery of knowledge components.
        
        Student ID: {student_id}
        Knowledge Component: {knowledge_component}
        Performance History: {performance_history}
        
        Please provide:
        1. Current mastery level estimation (0-1)
        2. Confidence in this estimate
        3. Recommended next steps for instruction
        """,
        parameters=["student_id", "knowledge_component", "performance_history"],
        example_inputs={
            "student_id": "S123",
            "knowledge_component": "Multiplication of fractions",
            "performance_history": "[[2023-01-01, 0.7], [2023-01-15, 0.8], [2023-02-01, 0.75]]"
        }
    )
    config.add_template(kt_template)
    
    return config

# Create singleton instance
dspy_config = create_default_config()

# Try to load from file if exists
config_path = os.path.join("config", "dspy_config.json")
if os.path.exists(config_path):
    loaded_config = DSPyConfig.load(config_path)
    if loaded_config:
        dspy_config = loaded_config
        logger.debug("Loaded DSPy configuration from file")
    else:
        # Save default config
        dspy_config.save(config_path)
        logger.debug("Created default DSPy configuration file")