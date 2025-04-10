#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Active Learning Interface for UI detection.
This module allows the detection system to actively ask questions when uncertain,
maximizing learning efficiency from user feedback.
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

class ActiveLearningInterface:
    """
    An interface for active learning that identifies uncertain detections and
    requests user input to maximize learning efficiency.
    """
    
    def __init__(self, feedback_callback: Callable = None, memory_path: str = None):
        """Initialize the active learning interface.
        
        Args:
            feedback_callback: Optional callback function to handle user feedback
            memory_path: Path to store active learning data
        """
        self.name = "ActiveLearningInterface"
        
        # Set up paths
        self.memory_path = Path(memory_path) if memory_path else Path('active_learning_memory')
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Set up feedback callback
        self.feedback_callback = feedback_callback
        
        # Memory for learning history
        self.questions_asked = []
        self.feedback_received = []
        self.uncertainty_thresholds = {
            'button': 0.65,
            'text_input': 0.70,
            'checkbox': 0.75,
            'icon': 0.60,
            'dropdown': 0.70,
            'toggle': 0.75
        }
        
        # Load learning history
        self.load_history()
        
        logger.info(f"Initialized active learning interface with {len(self.questions_asked)} previous questions")
    
    def load_history(self) -> None:
        """Load learning history from memory path."""
        history_file = self.memory_path / 'learning_history.json'
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    self.questions_asked = history.get('questions_asked', [])
                    self.feedback_received = history.get('feedback_received', [])
                    thresholds = history.get('uncertainty_thresholds', {})
                    if thresholds:
                        self.uncertainty_thresholds.update(thresholds)
                logger.info(f"Loaded active learning history with {len(self.questions_asked)} items")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading learning history: {e}")
    
    def save_history(self) -> None:
        """Save learning history to memory path."""
        history_file = self.memory_path / 'learning_history.json'
        
        try:
            history = {
                'questions_asked': self.questions_asked,
                'feedback_received': self.feedback_received,
                'uncertainty_thresholds': self.uncertainty_thresholds,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Saved active learning history with {len(self.questions_asked)} items")
        except IOError as e:
            logger.error(f"Error saving learning history: {e}")
    
    def process_detection_results(self, detection_results: List[Dict], 
                                 image: np.ndarray,
                                 max_questions: int = 3) -> List[Dict]:
        """Process detection results to identify uncertain detections for active learning.
        
        Args:
            detection_results: List of detected UI elements
            image: The image being analyzed
            max_questions: Maximum number of questions to ask
            
        Returns:
            List of uncertain detections to ask about
        """
        if not detection_results or not image.any():
            return []
            
        # Group detections by element type
        detections_by_type = {}
        for detection in detection_results:
            element_type = detection.get('type', 'unknown')
            if element_type not in detections_by_type:
                detections_by_type[element_type] = []
            detections_by_type[element_type].append(detection)
        
        # Find uncertain detections
        uncertain_detections = []
        
        for element_type, detections in detections_by_type.items():
            threshold = self.uncertainty_thresholds.get(element_type, 0.7)
            
            # Sort by confidence (ascending)
            detections.sort(key=lambda x: x.get('confidence', 0))
            
            # Consider the least confident detections first
            for detection in detections:
                confidence = detection.get('confidence', 0)
                
                # If confidence is below the uncertainty threshold, add to uncertain list
                if confidence < threshold:
                    # Extract the element image for review
                    rect = detection.get('rect', {})
                    if rect:
                        x, y = rect.get('x', 0), rect.get('y', 0)
                        w, h = rect.get('width', 10), rect.get('height', 10)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, image.shape[1] - 1))
                        y = max(0, min(y, image.shape[0] - 1))
                        w = min(w, image.shape[1] - x)
                        h = min(h, image.shape[0] - y)
                        
                        # Extract element image
                        if w > 0 and h > 0:
                            element_img = image[y:y+h, x:x+w]
                            
                            # Save the image for questioning
                            question_id = f"question_{len(self.questions_asked) + len(uncertain_detections)}"
                            img_path = self.memory_path / f"{question_id}.png"
                            cv2.imwrite(str(img_path), element_img)
                            
                            # Add to uncertain detections
                            uncertain_detections.append({
                                'question_id': question_id,
                                'detection': detection,
                                'image_path': str(img_path),
                                'timestamp': datetime.now().isoformat()
                            })
        
        # Limit to maximum number of questions
        uncertain_detections = uncertain_detections[:max_questions]
        
        # Record the questions being asked
        for uncertain in uncertain_detections:
            self.questions_asked.append({
                'question_id': uncertain['question_id'],
                'detection': uncertain['detection'],
                'image_path': uncertain['image_path'],
                'timestamp': uncertain['timestamp'],
                'status': 'pending'
            })
        
        # Save updated history
        self.save_history()
        
        return uncertain_detections
    
    def formulate_questions(self, uncertain_detections: List[Dict]) -> List[Dict]:
        """Formulate questions about uncertain detections.
        
        Args:
            uncertain_detections: List of uncertain detections
            
        Returns:
            List of formatted questions to present to the user
        """
        questions = []
        
        for i, uncertain in enumerate(uncertain_detections):
            detection = uncertain['detection']
            element_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 0)
            
            # Formulate question based on element type
            if element_type == 'button':
                question_text = f"Is this a button? (confidence: {confidence:.2f})"
            elif element_type == 'text_input':
                question_text = f"Is this a text input field? (confidence: {confidence:.2f})"
            elif element_type == 'checkbox':
                question_text = f"Is this a checkbox? (confidence: {confidence:.2f})"
            else:
                question_text = f"Is this a {element_type}? (confidence: {confidence:.2f})"
                
            questions.append({
                'question_id': uncertain['question_id'],
                'question_text': question_text,
                'element_type': element_type,
                'image_path': uncertain['image_path'],
                'options': ['Yes', 'No', 'It\'s something else']
            })
            
        return questions
    
    def process_feedback(self, question_id: str, feedback: str, 
                        alternative_type: str = None) -> None:
        """Process user feedback for an uncertain detection.
        
        Args:
            question_id: ID of the question being answered
            feedback: User feedback ('Yes', 'No', or 'It's something else')
            alternative_type: Alternative element type if 'It's something else'
        """
        # Find the question in the history
        question_index = None
        for i, question in enumerate(self.questions_asked):
            if question['question_id'] == question_id:
                question_index = i
                break
                
        if question_index is None:
            logger.warning(f"Question ID {question_id} not found in history")
            return
            
        # Get the original question
        question = self.questions_asked[question_index]
        detection = question['detection']
        element_type = detection.get('type', 'unknown')
        
        # Record feedback
        self.feedback_received.append({
            'question_id': question_id,
            'original_type': element_type,
            'feedback': feedback,
            'alternative_type': alternative_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update question status
        self.questions_asked[question_index]['status'] = 'answered'
        
        # Update uncertainty threshold based on feedback
        if feedback == 'Yes':
            # Correct detection despite low confidence, slightly lower threshold
            self.uncertainty_thresholds[element_type] = max(0.5, self.uncertainty_thresholds.get(element_type, 0.7) - 0.02)
        elif feedback == 'No':
            # Incorrect detection, increase threshold
            self.uncertainty_thresholds[element_type] = min(0.9, self.uncertainty_thresholds.get(element_type, 0.7) + 0.05)
        
        # Trigger feedback callback if available
        if self.feedback_callback:
            try:
                self.feedback_callback(question_id, feedback, alternative_type)
            except Exception as e:
                logger.error(f"Error in feedback callback: {e}")
        
        # Save updated history
        self.save_history()
        
        logger.info(f"Processed feedback for question {question_id}: {feedback}")
    
    def get_learning_efficiency(self) -> Dict:
        """Calculate the learning efficiency of the active learning process.
        
        Returns:
            Dictionary with learning efficiency metrics
        """
        # Count questions by type
        questions_by_type = {}
        for question in self.questions_asked:
            element_type = question['detection'].get('type', 'unknown')
            if element_type not in questions_by_type:
                questions_by_type[element_type] = {
                    'total': 0,
                    'answered': 0,
                    'correct': 0
                }
            
            questions_by_type[element_type]['total'] += 1
            
            if question.get('status') == 'answered':
                questions_by_type[element_type]['answered'] += 1
        
        # Count correct answers
        for feedback in self.feedback_received:
            element_type = feedback.get('original_type', 'unknown')
            if element_type in questions_by_type and feedback.get('feedback') == 'Yes':
                questions_by_type[element_type]['correct'] += 1
        
        # Calculate overall metrics
        total_questions = len(self.questions_asked)
        answered_questions = sum(1 for q in self.questions_asked if q.get('status') == 'answered')
        correct_answers = sum(1 for f in self.feedback_received if f.get('feedback') == 'Yes')
        
        # Calculate precision for each element type
        precision_by_type = {}
        for element_type, stats in questions_by_type.items():
            if stats['answered'] > 0:
                precision_by_type[element_type] = stats['correct'] / stats['answered']
            else:
                precision_by_type[element_type] = 0
        
        # Calculate overall precision
        overall_precision = correct_answers / answered_questions if answered_questions > 0 else 0
        
        return {
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'correct_answers': correct_answers,
            'response_rate': answered_questions / total_questions if total_questions > 0 else 0,
            'overall_precision': overall_precision,
            'precision_by_type': precision_by_type,
            'uncertainty_thresholds': self.uncertainty_thresholds
        }
    
    def extract_learning_insights(self) -> List[str]:
        """Extract insights from the active learning process.
        
        Returns:
            List of learning insights as text
        """
        insights = []
        
        # Get learning efficiency metrics
        efficiency = self.get_learning_efficiency()
        
        # Generate insights
        if efficiency['total_questions'] > 0:
            insights.append(f"Active learning has asked {efficiency['total_questions']} questions, with a {efficiency['response_rate']*100:.1f}% response rate.")
            
            if efficiency['answered_questions'] > 0:
                insights.append(f"Overall detection precision: {efficiency['overall_precision']*100:.1f}%")
                
                # Identify problematic element types
                problem_types = []
                for element_type, precision in efficiency['precision_by_type'].items():
                    if precision < 0.7:
                        problem_types.append(f"{element_type} ({precision*100:.1f}%)")
                        
                if problem_types:
                    insights.append(f"Element types needing improvement: {', '.join(problem_types)}")
                
                # Identify good element types
                good_types = []
                for element_type, precision in efficiency['precision_by_type'].items():
                    if precision > 0.9:
                        good_types.append(f"{element_type} ({precision*100:.1f}%)")
                        
                if good_types:
                    insights.append(f"Element types with good detection: {', '.join(good_types)}")
                    
            # Threshold insights
            high_thresholds = []
            low_thresholds = []
            for element_type, threshold in efficiency['uncertainty_thresholds'].items():
                if threshold > 0.8:
                    high_thresholds.append(f"{element_type} ({threshold:.2f})")
                elif threshold < 0.6:
                    low_thresholds.append(f"{element_type} ({threshold:.2f})")
                    
            if high_thresholds:
                insights.append(f"High uncertainty thresholds: {', '.join(high_thresholds)} - these types need more accurate detection")
                
            if low_thresholds:
                insights.append(f"Low uncertainty thresholds: {', '.join(low_thresholds)} - these types have reliable detection")
                
        else:
            insights.append("No active learning questions have been asked yet.")
            
        return insights
    
    def get_pending_questions(self) -> List[Dict]:
        """Get all pending questions that haven't been answered yet.
        
        Returns:
            List of pending questions
        """
        pending = []
        
        for question in self.questions_asked:
            if question.get('status') == 'pending':
                pending.append({
                    'question_id': question['question_id'],
                    'detection': question['detection'],
                    'image_path': question['image_path'],
                    'timestamp': question['timestamp']
                })
                
        return pending
    
    def apply_learning_to_detector(self, detector: Any) -> None:
        """Apply learning from user feedback to improve a detector.
        
        Args:
            detector: The detector to improve with learning
        """
        # Check if detector supports incremental learning
        if not hasattr(detector, 'supports_incremental_learning') or not detector.supports_incremental_learning():
            logger.warning(f"Detector {detector.name if hasattr(detector, 'name') else 'Unknown'} doesn't support incremental learning")
            return
            
        # Process feedback to train the detector
        correct_count = 0
        incorrect_count = 0
        
        for feedback in self.feedback_received:
            question_id = feedback['question_id']
            feedback_value = feedback['feedback']
            
            # Find the corresponding question
            question = None
            for q in self.questions_asked:
                if q['question_id'] == question_id:
                    question = q
                    break
                    
            if not question:
                continue
                
            # Get the image path
            image_path = question.get('image_path')
            if not image_path or not os.path.exists(image_path):
                continue
                
            try:
                # Load the element image
                element_img = cv2.imread(image_path)
                if element_img is None:
                    continue
                    
                # Get the detection info
                detection = question['detection']
                element_type = detection.get('type', 'unknown')
                
                # Process based on feedback
                if feedback_value == 'Yes':
                    # Correct detection
                    if hasattr(detector, 'add_training_example'):
                        detector.add_training_example(element_img, element_type, is_positive=True)
                    elif hasattr(detector, 'add_correct_detection'):
                        detector.add_correct_detection(element_img, detection)
                    correct_count += 1
                    
                elif feedback_value == 'No' or feedback_value == "It's something else":
                    # Incorrect detection
                    alternative = feedback.get('alternative_type')
                    
                    if hasattr(detector, 'add_training_example'):
                        # This was not the right element type
                        detector.add_training_example(element_img, element_type, is_positive=False)
                        
                        # If an alternative was provided, add as positive for that type
                        if alternative:
                            detector.add_training_example(element_img, alternative, is_positive=True)
                    incorrect_count += 1
                    
            except Exception as e:
                logger.error(f"Error applying learning to detector: {e}")
                
        logger.info(f"Applied active learning feedback to detector: {correct_count} correct and {incorrect_count} incorrect examples")
