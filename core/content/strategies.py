# core/content/strategies.py

"""Normalization strategies for different content types."""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Union, Type
from config.logging_config import get_module_logger
from core.schemas.content_schema import (
    EducationalContent, IEPContent, LessonPlanContent,
    ContentSection, ContentType
)

# Create a logger for this module
logger = get_module_logger("content_strategies")

# Type variable for content types
T = TypeVar('T', bound=EducationalContent)

class NormalizationStrategy(ABC):
    """Base class for content normalization strategies."""
    
    @abstractmethod
    def normalize(self, content: T) -> T:
        """Normalize content.
        
        Args:
            content: Content to normalize
            
        Returns:
            Normalized content
        """
        pass


class DefaultNormalizationStrategy(NormalizationStrategy):
    """Default normalization strategy for generic content."""
    
    def normalize(self, content: T) -> T:
        """Apply default normalization.
        
        Args:
            content: Content to normalize
            
        Returns:
            Normalized content
        """
        # Basic normalization only
        content = self._normalize_description(content)
        content = self._normalize_sections(content)
        
        return content
    
    def _normalize_description(self, content: T) -> T:
        """Normalize the content description.
        
        Args:
            content: Content to normalize
            
        Returns:
            Content with normalized description
        """
        # Ensure the description has proper capitalization and ending punctuation
        description = content.description.strip()
        
        if description:
            # Capitalize first letter
            description = description[0].upper() + description[1:]
            
            # Ensure ending punctuation
            if not description[-1] in '.!?':
                description += '.'
        
        content.description = description
        return content
    
    def _normalize_sections(self, content: T) -> T:
        """Normalize content sections.
        
        Args:
            content: Content to normalize
            
        Returns:
            Content with normalized sections
        """
        # Sort sections by order
        content.sections.sort(key=lambda s: s.order)
        
        # Ensure section titles are properly capitalized
        for section in content.sections:
            if section.title:
                section.title = section.title.strip()
                section.title = section.title[0].upper() + section.title[1:]
        
        return content


class IEPNormalizationStrategy(NormalizationStrategy):
    """Normalization strategy for IEP content."""
    
    def normalize(self, content: T) -> T:
        """Apply IEP-specific normalization.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            Normalized IEP content
        """
        if not isinstance(content, IEPContent):
            logger.warning(f"Content is not an IEP: {type(content).__name__}")
            return content
        
        # Apply generic normalizations
        content = self._normalize_description(content)
        content = self._normalize_sections(content)
        
        # IEP-specific normalizations
        content = self._normalize_goals(content)
        content = self._normalize_accommodations(content)
        content = self._normalize_services(content)
        
        return content
    
    def _normalize_description(self, content: IEPContent) -> IEPContent:
        """Normalize the IEP description.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            IEP with normalized description
        """
        # Ensure the description has proper capitalization and ending punctuation
        description = content.description.strip()
        
        if description:
            # Capitalize first letter
            description = description[0].upper() + description[1:]
            
            # Ensure ending punctuation
            if not description[-1] in '.!?':
                description += '.'
        
        content.description = description
        return content
    
    def _normalize_sections(self, content: IEPContent) -> IEPContent:
        """Normalize IEP sections.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            IEP with normalized sections
        """
        # Sort sections by order
        content.sections.sort(key=lambda s: s.order)
        
        # Ensure standard IEP sections are present
        required_sections = [
            "Student Information",
            "Present Levels of Performance",
            "Annual Goals",
            "Accommodations and Modifications",
            "Special Education Services",
            "Assessment Information"
        ]
        
        existing_section_titles = [s.title for s in content.sections]
        
        # Add missing sections
        for i, section_title in enumerate(required_sections):
            if section_title not in existing_section_titles:
                # Create new section
                new_section = ContentSection(
                    section_id=f"section_{i}",
                    title=section_title,
                    content="",
                    order=i,
                    content_type="text"
                )
                content.sections.append(new_section)
        
        # Sort sections by order
        content.sections.sort(key=lambda s: s.order)
        
        return content
    
    def _normalize_goals(self, content: IEPContent) -> IEPContent:
        """Normalize IEP goals.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            IEP with normalized goals
        """
        normalized_goals = []
        
        for goal in content.goals:
            norm_goal = dict(goal)
            
            # Ensure description has proper format
            if 'description' in norm_goal:
                desc = norm_goal['description'].strip()
                
                # Ensure starts with capital letter
                if desc:
                    desc = desc[0].upper() + desc[1:]
                
                # Ensure it ends with period
                if desc and not desc[-1] in '.!?':
                    desc += '.'
                
                norm_goal['description'] = desc
            
            normalized_goals.append(norm_goal)
        
        content.goals = normalized_goals
        return content
    
    def _normalize_accommodations(self, content: IEPContent) -> IEPContent:
        """Normalize IEP accommodations.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            IEP with normalized accommodations
        """
        normalized_accommodations = []
        
        for accommodation in content.accommodations:
            norm_acc = dict(accommodation)
            
            # Ensure description has proper format
            if 'description' in norm_acc:
                desc = norm_acc['description'].strip()
                
                # Ensure starts with capital letter
                if desc:
                    desc = desc[0].upper() + desc[1:]
                
                # Ensure it ends with period
                if desc and not desc[-1] in '.!?':
                    desc += '.'
                
                norm_acc['description'] = desc
            
            normalized_accommodations.append(norm_acc)
        
        content.accommodations = normalized_accommodations
        return content
    
    def _normalize_services(self, content: IEPContent) -> IEPContent:
        """Normalize IEP services.
        
        Args:
            content: IEP content to normalize
            
        Returns:
            IEP with normalized services
        """
        normalized_services = []
        
        for service in content.services:
            norm_service = dict(service)
            
            # Ensure description has proper format
            if 'description' in norm_service:
                desc = norm_service['description'].strip()
                
                # Ensure starts with capital letter
                if desc:
                    desc = desc[0].upper() + desc[1:]
                
                # Ensure it ends with period
                if desc and not desc[-1] in '.!?':
                    desc += '.'
                
                norm_service['description'] = desc
            
            normalized_services.append(norm_service)
        
        content.services = normalized_services
        return content


class LessonPlanNormalizationStrategy(NormalizationStrategy):
    """Normalization strategy for lesson plan content."""
    
    def normalize(self, content: T) -> T:
        """Apply lesson plan-specific normalization.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Normalized lesson plan content
        """
        if not isinstance(content, LessonPlanContent):
            logger.warning(f"Content is not a lesson plan: {type(content).__name__}")
            return content
        
        # Apply generic normalizations
        content = self._normalize_description(content)
        content = self._normalize_sections(content)
        
        # Lesson plan-specific normalizations
        content = self._normalize_objectives(content)
        content = self._normalize_materials(content)
        content = self._normalize_activities(content)
        
        return content
    
    def _normalize_description(self, content: LessonPlanContent) -> LessonPlanContent:
        """Normalize the lesson plan description.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Lesson plan with normalized description
        """
        # Ensure the description has proper capitalization and ending punctuation
        description = content.description.strip()
        
        if description:
            # Capitalize first letter
            description = description[0].upper() + description[1:]
            
            # Ensure ending punctuation
            if not description[-1] in '.!?':
                description += '.'
        
        content.description = description
        return content
    
    def _normalize_sections(self, content: LessonPlanContent) -> LessonPlanContent:
        """Normalize lesson plan sections.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Lesson plan with normalized sections
        """
        # Sort sections by order
        content.sections.sort(key=lambda s: s.order)
        
        # Ensure standard lesson plan sections are present
        required_sections = [
            "Objectives",
            "Materials",
            "Instructional Procedures",
            "Activities",
            "Assessment",
            "Closure"
        ]
        
        existing_section_titles = [s.title for s in content.sections]
        
        # Add missing sections
        for i, section_title in enumerate(required_sections):
            if section_title not in existing_section_titles:
                # Create new section
                new_section = ContentSection(
                    section_id=f"section_{i}",
                    title=section_title,
                    content="",
                    order=i,
                    content_type="text"
                )
                content.sections.append(new_section)
        
        # Sort sections by order
        content.sections.sort(key=lambda s: s.order)
        
        return content
    
    def _normalize_objectives(self, content: LessonPlanContent) -> LessonPlanContent:
        """Normalize lesson plan objectives.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Lesson plan with normalized objectives
        """
        normalized_objectives = []
        
        for objective in content.objectives:
            obj = objective.strip()
            
            # Ensure starts with capital letter
            if obj:
                obj = obj[0].upper() + obj[1:]
            
            # Ensure it ends with period
            if obj and not obj[-1] in '.!?':
                obj += '.'
            
            normalized_objectives.append(obj)
        
        content.objectives = normalized_objectives
        return content
    
    def _normalize_materials(self, content: LessonPlanContent) -> LessonPlanContent:
        """Normalize lesson plan materials.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Lesson plan with normalized materials
        """
        normalized_materials = []
        
        for material in content.materials:
            mat = material.strip()
            
            # Ensure starts with capital letter
            if mat:
                mat = mat[0].upper() + mat[1:]
            
            normalized_materials.append(mat)
        
        content.materials = normalized_materials
        return content
    
    def _normalize_activities(self, content: LessonPlanContent) -> LessonPlanContent:
        """Normalize lesson plan activities.
        
        Args:
            content: Lesson plan content to normalize
            
        Returns:
            Lesson plan with normalized activities
        """
        normalized_activities = []
        
        for activity in content.activities:
            norm_activity = dict(activity)
            
            # Ensure description has proper format
            if 'description' in norm_activity:
                desc = norm_activity['description'].strip()
                
                # Ensure starts with capital letter
                if desc:
                    desc = desc[0].upper() + desc[1:]
                
                # Ensure it ends with period
                if desc and not desc[-1] in '.!?':
                    desc += '.'
                
                norm_activity['description'] = desc
            
            normalized_activities.append(norm_activity)
        
        content.activities = normalized_activities
        return content