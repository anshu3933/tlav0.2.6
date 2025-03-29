# core/llm/dspy/layer.py

"""Implementation of DSPy layer for knowledge tracing."""

from typing import Dict, List, Any, Optional, Union, Callable, Type
import math
import numpy as np
from datetime import datetime
import importlib
from dataclasses import dataclass, field
from config.logging_config import get_module_logger
from core.schemas.student_schema import KnowledgeState, KnowledgeComponent

# Create a logger for this module
logger = get_module_logger("dspy_layer")

# Check if DSPy is available
try:
    dspy = importlib.import_module("dspy")
    DSPY_AVAILABLE = True
    logger.debug("DSPy is available")
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy is not available. Knowledge tracing layer will use fallback methods.")
    
    # Create dummy class for when DSPy is not available
    class DummyClass:
        """Dummy class for when DSPy is not available."""
        def __init__(self, *args, **kwargs):
            pass
    
    # Create dummy dspy module
    class DummyDSPy:
        """Dummy DSPy module."""
        Module = DummyClass
    
    dspy = DummyDSPy()

@dataclass
class AssessmentResult:
    """Result of a student assessment for knowledge tracing."""
    student_id: str
    kc_id: str
    score: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class BayesianKnowledgeTracing:
    """BKT algorithm for knowledge tracing."""
    
    def __init__(self, 
                 prior: float = 0.0,
                 learn: float = 0.1, 
                 slip: float = 0.1, 
                 guess: float = 0.2):
        """Initialize with BKT parameters.
        
        Args:
            prior: Prior probability of mastery
            learn: Probability of learning
            slip: Probability of slipping
            guess: Probability of guessing
        """
        self.prior = prior
        self.learn = learn
        self.slip = slip
        self.guess = guess
    
    def update(self, mastery: float, correct: bool) -> float:
        """Update mastery estimate based on assessment result.
        
        Args:
            mastery: Current mastery estimate
            correct: Whether the assessment was correct
            
        Returns:
            Updated mastery estimate
        """
        # Step 1: Adjust for slipping and guessing to get posterior
        if correct:
            posterior = (mastery * (1 - self.slip)) / (mastery * (1 - self.slip) + (1 - mastery) * self.guess)
        else:
            posterior = (mastery * self.slip) / (mastery * self.slip + (1 - mastery) * (1 - self.guess))
        
        # Step 2: Adjust for learning
        updated_mastery = posterior + (1 - posterior) * self.learn
        
        # Ensure value is in range [0, 1]
        return max(0.0, min(1.0, updated_mastery))
    
    def predict(self, mastery: float) -> float:
        """Predict probability of correct response.
        
        Args:
            mastery: Current mastery estimate
            
        Returns:
            Probability of correct response
        """
        return mastery * (1 - self.slip) + (1 - mastery) * self.guess

class KnowledgeTracingLayer:
    """Layer for tracking knowledge states of students."""
    
    def __init__(self, use_dspy: bool = True):
        """Initialize the knowledge tracing layer.
        
        Args:
            use_dspy: Whether to use DSPy for enhanced tracing
        """
        self.use_dspy = use_dspy and DSPY_AVAILABLE
        
        # Initialize BKT models
        self.bkt_models: Dict[str, BayesianKnowledgeTracing] = {}
        
        # Initialize DSPy module if available
        if self.use_dspy:
            self._init_dspy_module()
        
        logger.debug(f"Initialized knowledge tracing layer (use_dspy={self.use_dspy})")
    
    def _init_dspy_module(self) -> None:
        """Initialize the DSPy module for knowledge tracing."""
        if not DSPY_AVAILABLE:
            return
        
        try:
            # Import the knowledge tracing signature
            from core.llm.dspy.signatures import KnowledgeTracingAnalysis
            
            # Create module
            self.kt_module = dspy.Module(KnowledgeTracingAnalysis)
            
            logger.debug("Initialized DSPy knowledge tracing module")
        except Exception as e:
            logger.error(f"Error initializing DSPy module: {str(e)}")
            self.use_dspy = False
    
    def update_knowledge_state(self, 
                              knowledge_state: KnowledgeState, 
                              assessment_result: AssessmentResult) -> KnowledgeState:
        """Update a knowledge state based on an assessment result.
        
        Args:
            knowledge_state: Current knowledge state
            assessment_result: Assessment result
            
        Returns:
            Updated knowledge state
        """
        # Create BKT model if not exists
        kc_id = knowledge_state.kc_id
        if kc_id not in self.bkt_models:
            self.bkt_models[kc_id] = BayesianKnowledgeTracing(
                prior=knowledge_state.prior,
                learn=knowledge_state.learn,
                slip=knowledge_state.slip,
                guess=knowledge_state.guess
            )
        
        bkt_model = self.bkt_models[kc_id]
        
        # Convert score to binary correct/incorrect
        correct = assessment_result.score >= 0.7  # Threshold for correct
        
        # Update mastery with BKT
        updated_mastery = bkt_model.update(knowledge_state.mastery, correct)
        
        # Add assessment to history
        knowledge_state.assessment_history.append({
            "timestamp": assessment_result.timestamp,
            "score": assessment_result.score,
            "correct": correct,
            "mastery_before": knowledge_state.mastery,
            "mastery_after": updated_mastery
        })
        
        # Update knowledge state
        knowledge_state.mastery = updated_mastery
        knowledge_state.last_updated = datetime.now().isoformat()
        
        # Try to enhance with DSPy if available
        if self.use_dspy:
            try:
                # Format performance history
                performance_history = [
                    [entry.get("timestamp", ""), entry.get("score", 0.0)]
                    for entry in knowledge_state.assessment_history
                ]
                
                # Use DSPy module for enhanced analysis
                result = self.kt_module(
                    student_id=assessment_result.student_id,
                    knowledge_component=kc_id,
                    performance_history=str(performance_history)
                )
                
                # Update with DSPy insights if available
                if hasattr(result, "mastery_level") and result.mastery_level:
                    try:
                        # Parse mastery value (should be a number between 0-1)
                        dspy_mastery = float(result.mastery_level)
                        
                        # Combine BKT and DSPy estimates with weighted average
                        # BKT gets higher weight to maintain stability
                        combined_mastery = 0.7 * knowledge_state.mastery + 0.3 * dspy_mastery
                        knowledge_state.mastery = max(0.0, min(1.0, combined_mastery))
                        
                        logger.debug(f"Enhanced mastery with DSPy: {knowledge_state.mastery}")
                    except ValueError:
                        # If parsing fails, keep the BKT estimate
                        pass
                
                # Update confidence if available
                if hasattr(result, "confidence") and result.confidence:
                    try:
                        knowledge_state.confidence = float(result.confidence)
                    except ValueError:
                        pass
                
                # Update learning rate if available
                if hasattr(result, "learning_rate") and result.learning_rate:
                    try:
                        new_learn_rate = float(result.learning_rate)
                        knowledge_state.learn = max(0.01, min(0.5, new_learn_rate))
                    except ValueError:
                        pass
                
            except Exception as e:
                logger.error(f"Error enhancing knowledge state with DSPy: {str(e)}")
        
        return knowledge_state
    
    def predict_performance(self, knowledge_state: KnowledgeState) -> float:
        """Predict performance on next assessment.
        
        Args:
            knowledge_state: Current knowledge state
            
        Returns:
            Predicted probability of correct response
        """
        # Create BKT model if not exists
        kc_id = knowledge_state.kc_id
        if kc_id not in self.bkt_models:
            self.bkt_models[kc_id] = BayesianKnowledgeTracing(
                prior=knowledge_state.prior,
                learn=knowledge_state.learn,
                slip=knowledge_state.slip,
                guess=knowledge_state.guess
            )
        
        bkt_model = self.bkt_models[kc_id]
        
        # Predict using BKT
        return bkt_model.predict(knowledge_state.mastery)
    
    def analyze_learning_curve(self, knowledge_state: KnowledgeState) -> Dict[str, Any]:
        """Analyze the learning curve for a knowledge component.
        
        Args:
            knowledge_state: Knowledge state with assessment history
            
        Returns:
            Analysis results
        """
        history = knowledge_state.assessment_history
        
        if len(history) < 2:
            return {
                "learning_rate": knowledge_state.learn,
                "mastery_trend": "insufficient_data",
                "stability": 0.0,
                "prediction": knowledge_state.mastery
            }
        
        # Extract mastery values and timestamps
        mastery_values = [entry.get("mastery_after", 0.0) for entry in history]
        timestamps = [datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat())) 
                     for entry in history]
        
        # Calculate learning rate (average change in mastery per assessment)
        mastery_changes = [mastery_values[i] - mastery_values[i-1] for i in range(1, len(mastery_values))]
        avg_learning_rate = sum(mastery_changes) / len(mastery_changes)
        
        # Calculate time-based learning rate
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 86400  # Convert to days
                          for i in range(1, len(timestamps))]
            
            # Calculate learning rate per day
            daily_changes = [change / max(0.1, diff) for change, diff in zip(mastery_changes, time_diffs)]
            avg_daily_rate = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        else:
            avg_daily_rate = 0
        
        # Determine mastery trend
        recent_values = mastery_values[-min(3, len(mastery_values)):]
        if all(recent_values[i] >= recent_values[i-1] for i in range(1, len(recent_values))):
            trend = "increasing"
        elif all(recent_values[i] <= recent_values[i-1] for i in range(1, len(recent_values))):
            trend = "decreasing"
        else:
            trend = "fluctuating"
        
        # Calculate stability (inverse of variance)
        if len(recent_values) > 1:
            variance = np.var(recent_values) if np else sum((x - sum(recent_values)/len(recent_values))**2 for x in recent_values) / len(recent_values)
            stability = max(0.0, 1.0 - min(1.0, variance * 10))  # Scale to 0-1
        else:
            stability = 1.0
        
        # Make prediction for next assessment
        prediction = self.predict_performance(knowledge_state)
        
        return {
            "learning_rate": avg_learning_rate,
            "daily_learning_rate": avg_daily_rate,
            "mastery_trend": trend,
            "stability": stability,
            "prediction": prediction,
            "confidence": knowledge_state.confidence
        }
    
    def recommend_activities(self, 
                            knowledge_state: KnowledgeState,
                            knowledge_component: KnowledgeComponent) -> List[Dict[str, Any]]:
        """Recommend learning activities based on knowledge state.
        
        Args:
            knowledge_state: Current knowledge state
            knowledge_component: Knowledge component information
            
        Returns:
            List of recommended activities
        """
        mastery = knowledge_state.mastery
        
        # Basic recommendations based on mastery level
        if mastery < 0.3:
            difficulty = "introductory"
            focus = "foundation"
        elif mastery < 0.7:
            difficulty = "practice"
            focus = "reinforcement"
        else:
            difficulty = "advanced"
            focus = "mastery"
        
        # Check for prerequisites
        prerequisites = knowledge_component.prerequisites
        missing_prerequisites = []
        
        # Try to get enhanced recommendations with DSPy
        if self.use_dspy:
            try:
                # Format performance history
                performance_history = [
                    [entry.get("timestamp", ""), entry.get("score", 0.0)]
                    for entry in knowledge_state.assessment_history
                ]
                
                # Use DSPy module for recommendations
                result = self.kt_module(
                    student_id="",  # Not needed for recommendations
                    knowledge_component=knowledge_component.name,
                    performance_history=str(performance_history)
                )
                
                # Parse recommendations if available
                if hasattr(result, "recommendations") and result.recommendations:
                    # Try to parse as list or string
                    recommendations_text = result.recommendations
                    
                    try:
                        # Try parsing as JSON
                        dspy_recommendations = json.loads(recommendations_text)
                        if isinstance(dspy_recommendations, list):
                            return dspy_recommendations
                    except:
                        # Parse as text
                        recs = recommendations_text.split("\n")
                        parsed_recs = []
                        
                        for rec in recs:
                            if not rec.strip():
                                continue
                            
                            parsed_recs.append({
                                "description": rec.strip(),
                                "difficulty": difficulty,
                                "focus": focus,
                                "type": "practice"
                            })
                        
                        if parsed_recs:
                            return parsed_recs
                
            except Exception as e:
                logger.error(f"Error getting DSPy recommendations: {str(e)}")
        
        # Fallback to standard recommendations
        recommendations = [
            {
                "type": "practice",
                "difficulty": difficulty,
                "focus": focus,
                "description": f"{difficulty.capitalize()} practice for {knowledge_component.name}"
            }
        ]
        
        # Add prerequisite work if needed
        if mastery < 0.5 and prerequisites:
            for prereq in prerequisites[:2]:  # Limit to top 2 prerequisites
                recommendations.append({
                    "type": "prerequisite",
                    "difficulty": "practice",
                    "focus": "foundation",
                    "description": f"Review prerequisite: {prereq}"
                })
        
        # Add advanced work for high mastery
        if mastery > 0.8:
            recommendations.append({
                "type": "extension",
                "difficulty": "advanced",
                "focus": "application",
                "description": f"Apply {knowledge_component.name} in new contexts"
            })
        
        return recommendations

# Create singleton instance
knowledge_tracing_layer = KnowledgeTracingLayer()