"""
Clarification Engine

This module provides clarification capabilities when the system is uncertain about
automation actions. It generates context-aware questions, handles user responses,
and learns from these interactions to improve future automation.

The clarification engine serves as a critical bridge between autonomous operation
and user oversight, enabling NEXUS to learn from interactions rather than strictly
following predefined rules.
"""

import time
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class ClarificationEngine:
    """
    Engine for generating clarification questions when uncertainty arises
    during automation tasks.
    
    The engine analyzes the current context, generates appropriate questions,
    processes user responses, and improves future performance by learning
    from these interactions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the clarification engine.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - confidence_threshold: Minimum confidence level before asking for clarification (default: 0.7)
                - max_attempts: Maximum number of clarification attempts (default: 3)
                - learning_rate: Rate at which the system incorporates feedback (default: 0.2)
                - memory_path: Path to store clarification history (default: None)
                - question_templates: Custom question templates (default: None)
                - enable_learning: Whether to learn from interactions (default: True)
        """
        self.config = config or {}
        
        # Configuration settings
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.max_attempts = self.config.get("max_attempts", 3)
        self.learning_rate = self.config.get("learning_rate", 0.2)
        self.memory_path = self.config.get("memory_path")
        self.enable_learning = self.config.get("enable_learning", True)
        
        # Callback for getting responses to questions
        self._response_callback = None
        
        # History of clarifications
        self.clarification_history = []
        
        # Question templates for different scenarios
        self.question_templates = self.config.get("question_templates") or {
            "ui_element_action": [
                "I see a {element_type} labeled '{element_text}'. Should I {action} it?",
                "There's a {element_type} that appears to say '{element_text}'. Do you want me to {action} it?",
                "I've detected a {element_type} with text '{element_text}'. Would you like me to {action} it?"
            ],
            "ui_element_ambiguous": [
                "I found multiple {element_type}s. Which one should I interact with: {options}?",
                "There are several {element_type}s visible. Which one do you want: {options}?",
                "I see {count} {element_type}s. Which one should I select: {options}?"
            ],
            "text_input": [
                "I need to enter text in a {element_type}. What should I type?",
                "There's a {element_type} field. What text would you like me to enter?",
                "I see a {element_type} where I can input text. What should I write here?"
            ],
            "navigation": [
                "I need to navigate to {destination}. Is this correct?",
                "Should I navigate to {destination}?",
                "Do you want me to go to {destination}?"
            ],
            "dangerous_action": [
                "The action {action} seems potentially risky. Should I proceed anyway?",
                "I'm about to {action}, which could have significant effects. Are you sure?",
                "Caution: {action} might be dangerous. Do you want me to continue?"
            ],
            "general_uncertainty": [
                "I'm uncertain about what to do next. Should I {action}?",
                "I'm not confident about the next step. Would it be correct to {action}?",
                "I'm hesitant about {action}. Is this what you want me to do?"
            ]
        }
        
        # Load existing clarification history if available
        if self.memory_path:
            self._load_history()
    
    def set_response_callback(self, callback: Callable[[str], str]):
        """
        Set the callback function that will be used to get responses to clarification questions.
        
        Args:
            callback: A function that takes a question string and returns a response string
        """
        self._response_callback = callback
    
    def needs_clarification(self, confidence: float) -> bool:
        """
        Determine if clarification is needed based on confidence level.
        
        Args:
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            True if clarification is needed, False otherwise
        """
        return confidence < self.confidence_threshold
    
    def generate_question(self, scenario: str, context: Dict) -> str:
        """
        Generate a clarification question based on the scenario and context.
        
        Args:
            scenario: The type of scenario requiring clarification
            context: Context dictionary with information for the question
            
        Returns:
            A formatted question string
        """
        if scenario not in self.question_templates:
            scenario = "general_uncertainty"
            
        templates = self.question_templates[scenario]
        
        # Select a template, trying to vary the questions over time
        template_index = len(self.clarification_history) % len(templates)
        template = templates[template_index]
        
        try:
            # Format the template with the context
            question = template.format(**context)
            return question
        except KeyError as e:
            logger.warning(f"Missing context key for question template: {e}")
            # Fallback to a simple question
            return f"Should I proceed with this {context.get('action', 'action')}?"
    
    def ask_for_clarification(self, scenario: str, context: Dict, 
                              confidence: float = 0.0) -> Dict:
        """
        Ask for clarification about an uncertain action.
        
        Args:
            scenario: The type of scenario requiring clarification
            context: Context dictionary with information for the question
            confidence: Confidence level in the action (0.0 to 1.0)
            
        Returns:
            A dictionary with the clarification result:
                - question: The question that was asked
                - response: The user's response
                - proceed: Whether to proceed with the action
                - confidence: Updated confidence after clarification
                - clarification_id: Unique ID for this clarification
        """
        if not self._response_callback:
            logger.error("No response callback set for clarification engine")
            return {
                "question": None,
                "response": None,
                "proceed": False,
                "confidence": confidence,
                "clarification_id": None
            }
        
        # Generate question
        question = self.generate_question(scenario, context)
        
        # Ask the question
        response = self._response_callback(question)
        
        # Process the response
        proceed, updated_confidence = self._process_response(response, confidence)
        
        # Create clarification record
        clarification_id = f"clarification_{int(time.time())}_{len(self.clarification_history)}"
        clarification_record = {
            "id": clarification_id,
            "timestamp": time.time(),
            "scenario": scenario,
            "context": context,
            "question": question,
            "response": response,
            "initial_confidence": confidence,
            "updated_confidence": updated_confidence,
            "proceed": proceed
        }
        
        # Store in history
        self.clarification_history.append(clarification_record)
        
        # Save history if path provided
        if self.memory_path:
            self._save_history()
        
        # Learn from this interaction if enabled
        if self.enable_learning:
            self._learn_from_clarification(clarification_record)
        
        return {
            "question": question,
            "response": response,
            "proceed": proceed,
            "confidence": updated_confidence,
            "clarification_id": clarification_id
        }
    
    def _process_response(self, response: str, initial_confidence: float) -> Tuple[bool, float]:
        """
        Process a user response to determine if the action should proceed.
        
        Args:
            response: The user's response text
            initial_confidence: Initial confidence level
            
        Returns:
            Tuple of (proceed, updated_confidence)
        """
        # Normalize response
        response_lower = response.lower().strip()
        
        # Positive responses
        positive_indicators = ["yes", "yeah", "yep", "sure", "ok", "proceed", 
                               "correct", "right", "true", "confirm", "go ahead"]
        
        # Negative responses
        negative_indicators = ["no", "nope", "stop", "don't", "wait", "incorrect", 
                               "wrong", "false", "cancel", "negative"]
        
        # Check for positive response
        for indicator in positive_indicators:
            if indicator in response_lower:
                # Increase confidence significantly for explicit approval
                new_confidence = min(1.0, initial_confidence + 0.3)
                return True, new_confidence
        
        # Check for negative response
        for indicator in negative_indicators:
            if indicator in response_lower:
                # Decrease confidence for explicit rejection
                new_confidence = max(0.0, initial_confidence - 0.1)
                return False, new_confidence
        
        # If response is unclear, slightly favor proceeding but with lower confidence
        return True, initial_confidence + 0.05
    
    def _learn_from_clarification(self, clarification: Dict):
        """
        Learn from a clarification interaction to improve future behavior.
        
        Args:
            clarification: The clarification record dictionary
        """
        # Simple pattern recognition that can be enhanced with more sophisticated ML
        if not self.enable_learning:
            return
            
        scenario = clarification["scenario"]
        context = clarification["context"]
        proceed = clarification["proceed"]
        
        # Adjust confidence threshold based on historical patterns
        if len(self.clarification_history) > 10:
            # Calculate the rate of positive responses
            recent_history = self.clarification_history[-10:]
            positive_rate = sum(1 for c in recent_history if c["proceed"]) / len(recent_history)
            
            # If users mostly say yes, we can lower our threshold slightly
            if positive_rate > 0.8:
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
            # If users mostly say no, we should increase our threshold
            elif positive_rate < 0.3:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
    
    def get_similar_clarifications(self, context: Dict, limit: int = 5) -> List[Dict]:
        """
        Find similar clarification scenarios from history.
        
        Args:
            context: The current context dictionary
            limit: Maximum number of similar clarifications to return
            
        Returns:
            List of similar clarification records
        """
        if not self.clarification_history:
            return []
            
        # Calculate similarity scores
        scores = []
        for record in self.clarification_history:
            score = self._calculate_context_similarity(record["context"], context)
            scores.append((score, record))
        
        # Sort by similarity score (descending)
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return top matches
        return [record for _, record in scores[:limit]]
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """
        Calculate a similarity score between two context dictionaries.
        
        Args:
            context1: First context dictionary
            context2: Second context dictionary
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check for exact matches of key fields
        exact_match_keys = ["element_type", "action"]
        exact_matches = 0
        for key in exact_match_keys:
            if key in context1 and key in context2 and context1[key] == context2[key]:
                exact_matches += 1
                
        # Check for text similarity
        text_similarity = 0.0
        if "element_text" in context1 and "element_text" in context2:
            # Simple string similarity - could be enhanced with better algorithms
            text1 = context1["element_text"].lower()
            text2 = context2["element_text"].lower()
            
            # Check for substring match
            if text1 in text2 or text2 in text1:
                text_similarity = 0.8
            else:
                # Count common words
                words1 = set(text1.split())
                words2 = set(text2.split())
                common_words = words1.intersection(words2)
                if words1 or words2:  # Avoid division by zero
                    text_similarity = len(common_words) / max(len(words1), len(words2))
        
        # Weight factors
        exact_match_weight = 0.6
        text_similarity_weight = 0.4
        
        # Calculate overall similarity
        if exact_match_keys:
            exact_match_score = exact_matches / len(exact_match_keys)
        else:
            exact_match_score = 0
            
        overall_similarity = (exact_match_score * exact_match_weight + 
                             text_similarity * text_similarity_weight)
        
        return overall_similarity
    
    def get_clarification_statistics(self) -> Dict:
        """
        Get statistics about clarification history and learning.
        
        Returns:
            Dictionary with statistics
        """
        if not self.clarification_history:
            return {
                "total_clarifications": 0,
                "proceed_rate": 0,
                "current_confidence_threshold": self.confidence_threshold,
                "most_common_scenarios": {}
            }
            
        # Count by scenario type
        scenario_counts = {}
        for record in self.clarification_history:
            scenario = record["scenario"]
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        # Calculate proceed rate
        proceed_count = sum(1 for record in self.clarification_history if record["proceed"])
        proceed_rate = proceed_count / len(self.clarification_history)
        
        # Sort scenarios by count
        sorted_scenarios = sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_clarifications": len(self.clarification_history),
            "proceed_rate": proceed_rate,
            "current_confidence_threshold": self.confidence_threshold,
            "most_common_scenarios": dict(sorted_scenarios[:5])
        }
    
    def _save_history(self):
        """Save clarification history to disk"""
        if not self.memory_path:
            return
            
        # Create directory if it doesn't exist
        memory_path = Path(self.memory_path)
        memory_path.mkdir(parents=True, exist_ok=True)
        
        # Save history
        history_path = memory_path / "clarification_history.json"
        
        # Copy history and remove any non-serializable elements
        clean_history = []
        for record in self.clarification_history:
            clean_record = record.copy()
            # Clean context if needed
            if "context" in clean_record:
                clean_context = {}
                for k, v in clean_record["context"].items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        clean_context[k] = v
                clean_record["context"] = clean_context
            clean_history.append(clean_record)
            
        with open(history_path, 'w') as f:
            json.dump(clean_history, f, indent=2)
    
    def _load_history(self):
        """Load clarification history from disk"""
        if not self.memory_path:
            return
            
        history_path = Path(self.memory_path) / "clarification_history.json"
        if not history_path.exists():
            return
            
        try:
            with open(history_path, 'r') as f:
                self.clarification_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading clarification history: {e}")
            self.clarification_history = []
    
    def add_question_templates(self, scenario: str, templates: List[str]):
        """
        Add new question templates for a scenario.
        
        Args:
            scenario: Scenario identifier
            templates: List of template strings
        """
        if scenario in self.question_templates:
            self.question_templates[scenario].extend(templates)
        else:
            self.question_templates[scenario] = templates
    
    def get_example_clarification(self, scenario: str) -> Dict:
        """
        Generate an example clarification for a given scenario.
        
        Args:
            scenario: The scenario type
            
        Returns:
            An example clarification dictionary
        """
        # Default contexts for different scenarios
        example_contexts = {
            "ui_element_action": {
                "element_type": "button",
                "element_text": "Submit",
                "action": "click"
            },
            "ui_element_ambiguous": {
                "element_type": "button",
                "count": 3,
                "options": "Submit, Cancel, Help"
            },
            "text_input": {
                "element_type": "text field"
            },
            "navigation": {
                "destination": "Settings page"
            },
            "dangerous_action": {
                "action": "delete all files"
            },
            "general_uncertainty": {
                "action": "proceed to the next screen"
            }
        }
        
        # Use default context if available, otherwise use a generic one
        context = example_contexts.get(scenario, {"action": "perform this action"})
        
        # Generate question
        question = self.generate_question(scenario, context)
        
        return {
            "scenario": scenario,
            "context": context,
            "question": question,
            "example": True
        }
