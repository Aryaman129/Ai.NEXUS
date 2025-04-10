#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive UI Detection Test Script

This script demonstrates the advanced adaptive learning capabilities of the
enhanced UI detection system, including:

1. Self-expanding template library that learns from successful detections
2. Multi-API vision detection using Gemini, Hugging Face, OpenAI, or Groq 
3. Adaptive creation of specialized detectors based on detection patterns
4. Continuous improvement through feedback and learning

"""

import os
import sys
import json
import time
import uuid
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("adaptive_ui_detection")

# Add NEXUS to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import detector modules
try:
    # Create mock classes for testing in case imports fail
    class EnhancedDetectorRegistry:
        def __init__(self):
            self.detectors = {}
            
        def register_detector(self, detector, detector_id=None, name=None):
            detector_name = name or getattr(detector, 'name', f'detector_{len(self.detectors)}')
            detector_id = detector_id or detector_name
            self.detectors[detector_name] = detector
            logger.info(f"Registered detector: {detector_name}")
            
        def get_detector(self, detector_name):
            if detector_name in self.detectors:
                return self.detectors[detector_name]
            else:
                logger.warning(f"Detector {detector_name} not found")
                return None
            
        def detect_with_detector(self, detector_name, image, context=None):
            if detector_name in self.detectors:
                return self.detectors[detector_name].detect_elements(image, context)
            else:
                logger.warning(f"Detector {detector_name} not found")
                return []
    
    class MultimodalDetectionCoordinator:
        def __init__(self, detector_registry=None):
            self.registry = detector_registry or EnhancedDetectorRegistry()
            
        def detect_elements(self, image, context=None):
            results = []
            for detector_name, detector in self.registry.detectors.items():
                try:
                    elements = detector.detect_elements(image, context)
                    if elements:
                        results.extend(elements)
                except Exception as e:
                    logger.error(f"Error with detector {detector_name}: {e}")
            return results
    
    # Now try to import the real modules
    import importlib.util
    
    # Check if modules exist and import them
    for module_name in [
        'self_expanding_template_library', 
        'adaptive_detector_factory',
        'multi_api_vision_detector',
        'neural_network_detector',
        'cross_context_learner',
        'active_learning_interface'
    ]:
        full_path = os.path.join(
            os.path.dirname(__file__), 
            'src', 'ai_core', 'screen_analysis', 'ml_ui_detection', 
            f"{module_name}.py"
        )
        
        if os.path.exists(full_path):
            module_spec = importlib.util.spec_from_file_location(module_name, full_path)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            globals().update({name: getattr(module, name) for name in dir(module) 
                             if not name.startswith('_')})
        
    # Import function names directly into global namespace
    from src.ai_core.screen_analysis.ml_ui_detection.self_expanding_template_library import SelfExpandingTemplateLibrary
    from src.ai_core.screen_analysis.ml_ui_detection.adaptive_detector_factory import AdaptiveDetectorFactory
    from src.ai_core.screen_analysis.ml_ui_detection.multi_api_vision_detector import MultiAPIVisionDetector
    from src.ai_core.screen_analysis.ml_ui_detection.neural_network_detector import NeuralNetworkDetector
    from src.ai_core.screen_analysis.ml_ui_detection.cross_context_learner import CrossContextLearner
    from src.ai_core.screen_analysis.ml_ui_detection.active_learning_interface import ActiveLearningInterface
    
    logger.info("Successfully imported detector modules")
except Exception as e:
    logger.warning(f"Error importing real modules: {e}")
    logger.info("Continuing with mock implementations for testing")

# Create mock implementation of NeuralNetworkDetector for when TensorFlow isn't available
class MockNeuralNetworkDetector:
    """Mock implementation of NeuralNetworkDetector for when TensorFlow isn't available."""
    
    def __init__(self):
        self.name = "MockNeuralNetworkDetector"
        self.description = "Mock neural network detector that simulates learning"
        self.training_examples = []
        self.fine_tuning_sessions = 0
        logger.info("Created mock neural network detector (TensorFlow not available)")
    
    def detect_elements(self, image, context=None):
        """Simulate element detection with the neural network."""
        # Return some mock detections based on image size
        detected_elements = []
        
        if image is None or not isinstance(image, np.ndarray):
            return []
            
        height, width = image.shape[:2]
        
        # Create a simple mockup detection
        element_types = ['button', 'text_input', 'checkbox', 'icon']
        
        # Add a few mock detections
        for i in range(3):
            x = width // 4 + (i * 100)
            y = height // 3
            w = 80
            h = 30
            
            # Select element type
            element_type = element_types[i % len(element_types)]
            
            # Add detection with simulated confidence
            confidence = 0.5 + (0.1 * len(self.training_examples) / 10)  # Confidence improves with training
            confidence = min(0.95, confidence)  # Cap at 0.95
            
            detected_elements.append({
                'id': f"neural_{element_type}_{i}",
                'type': element_type,
                'rect': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                },
                'confidence': confidence,
                'text': f"{element_type.capitalize()} {i+1}",
                'source_detector': self.name
            })
        
        return detected_elements
    
    def add_training_example(self, image, element_type, is_positive=True):
        """Add a training example to simulate learning."""
        self.training_examples.append({
            'element_type': element_type,
            'is_positive': is_positive,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Added {'positive' if is_positive else 'negative'} training example for {element_type}")
    
    def supports_incremental_learning(self):
        """Whether this detector supports incremental learning."""
        return True
    
    def get_training_stats(self):
        """Get statistics about the neural network detector training."""
        return {
            'fine_tuning_sessions': self.fine_tuning_sessions,
            'training_examples_seen': len(self.training_examples),
            'positive_examples': sum(1 for ex in self.training_examples if ex['is_positive']),
            'negative_examples': sum(1 for ex in self.training_examples if not ex['is_positive'])
        }
    
    def add_correct_detection(self, image, element_data):
        """Add a correct detection example for fine-tuning."""
        element_type = element_data.get('type', '')
        if element_type:
            self.add_training_example(image, element_type, True)

# Create mock implementation of CrossContextLearner for testing
class MockCrossContextLearner:
    """Mock implementation of CrossContextLearner for testing without dependencies."""
    
    def __init__(self, memory_path=None):
        self.name = "MockCrossContextLearner"
        self.contexts = {}
        self.transfers_performed = 0
        self.knowledge_items_shared = 0
        logger.info("Created mock cross-context learner")
    
    def add_context_knowledge(self, context_name, element_data, screenshot=None):
        """Add element knowledge for a specific context."""
        if context_name not in self.contexts:
            self.contexts[context_name] = {
                'elements': [],
                'creation_date': datetime.now().isoformat()
            }
        
        # Make a copy of the element data
        element_copy = element_data.copy()
        element_copy['added_on'] = datetime.now().isoformat()
        
        self.contexts[context_name]['elements'].append(element_copy)
        logger.info(f"Added {element_data.get('type', 'unknown')} element to {context_name} context")
    
    def find_similar_elements(self, query_element, context_name=None):
        """Find elements similar to the query element."""
        similar_elements = []
        
        element_type = query_element.get('type', 'unknown')
        if element_type == 'unknown':
            return []
        
        # Search all contexts
        for ctx_name, context in self.contexts.items():
            if context_name and ctx_name != context_name:
                continue
                
            for element in context['elements']:
                if element.get('type') == element_type:
                    # Calculate a simple similarity score
                    similarity = 0.5
                    
                    # Same type is a strong indicator
                    if element.get('type') == query_element.get('type'):
                        similarity += 0.3
                        
                    # Similar text is a good indicator
                    if element.get('text', '').lower() == query_element.get('text', '').lower():
                        similarity += 0.2
                    
                    similar_elements.append({
                        'context': ctx_name,
                        'element': element,
                        'similarity': similarity,
                        'id': element.get('id', 'unknown')
                    })
        
        # Sort by similarity (highest first)
        similar_elements.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_elements[:5]  # Return top 5
    
    def transfer_knowledge(self, source_context, target_context, element_type=None):
        """Transfer knowledge from source context to target context."""
        if source_context not in self.contexts:
            return {'success': False, 'message': f"Source context {source_context} not found"}
            
        # Create target context if needed
        if target_context not in self.contexts:
            self.contexts[target_context] = {
                'elements': [],
                'creation_date': datetime.now().isoformat()
            }
            
        # Track statistics
        transferred_count = 0
        element_types_transferred = set()
        
        # Transfer elements
        for element in self.contexts[source_context]['elements']:
            elem_type = element.get('type', 'unknown')
            
            # Skip if element type filter is specified and doesn't match
            if element_type and elem_type != element_type:
                continue
                
            # Create a copy of the element
            element_copy = element.copy()
            element_copy['transferred_from'] = source_context
            element_copy['transferred_on'] = datetime.now().isoformat()
            
            # Add to target context
            self.contexts[target_context]['elements'].append(element_copy)
            
            transferred_count += 1
            element_types_transferred.add(elem_type)
            
        # Update statistics
        self.transfers_performed += 1
        self.knowledge_items_shared += transferred_count
        
        logger.info(f"Transferred {transferred_count} elements from {source_context} to {target_context}")
        
        return {
            'success': True,
            'stats': {
                'source': source_context,
                'target': target_context,
                'element_types_transferred': list(element_types_transferred),
                'elements_transferred': transferred_count
            }
        }
    
    def get_context_knowledge_stats(self):
        """Get statistics about the cross-context knowledge base."""
        context_stats = {}
        for context_name, context in self.contexts.items():
            element_types = {}
            for element in context['elements']:
                element_type = element.get('type', 'unknown')
                if element_type not in element_types:
                    element_types[element_type] = 0
                element_types[element_type] += 1
                
            context_stats[context_name] = {
                'element_count': len(context['elements']),
                'element_types': element_types,
                'creation_date': context.get('creation_date', '')
            }
            
        return {
            'context_count': len(self.contexts),
            'element_type_count': len(set(elem.get('type', 'unknown') for ctx in self.contexts.values() for elem in ctx['elements'])),
            'transfers_performed': self.transfers_performed,
            'knowledge_items_shared': self.knowledge_items_shared,
            'contexts': context_stats
        }

# Create mock implementation of ActiveLearningInterface for testing
class MockActiveLearningInterface:
    """Mock implementation of ActiveLearningInterface for testing without dependencies."""
    
    def __init__(self, memory_path=None, feedback_callback=None):
        self.name = "MockActiveLearningInterface"
        self.questions_asked = []
        self.feedback_received = []
        self.uncertainty_thresholds = {
            'button': 0.65,
            'text_input': 0.70,
            'checkbox': 0.75,
            'icon': 0.60
        }
        logger.info("Created mock active learning interface")
    
    def process_detection_results(self, detection_results, image, max_questions=3):
        """Process detection results to identify uncertain detections for active learning."""
        if not detection_results or image is None:
            return []
            
        uncertain_detections = []
        
        # Find elements with confidence below threshold
        for detection in detection_results:
            element_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 1.0)
            threshold = self.uncertainty_thresholds.get(element_type, 0.7)
            
            if confidence < threshold:
                question_id = f"question_{len(self.questions_asked) + len(uncertain_detections)}"
                
                uncertain_detections.append({
                    'question_id': question_id,
                    'detection': detection,
                    'image_path': "mock_path.png",  # Mock path for testing
                    'timestamp': datetime.now().isoformat()
                })
                
                # Track questions
                self.questions_asked.append({
                    'question_id': question_id,
                    'detection': detection,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'pending'
                })
                
                # Limit questions
                if len(uncertain_detections) >= max_questions:
                    break
        
        logger.info(f"Identified {len(uncertain_detections)} uncertain detections for active learning")
        return uncertain_detections
    
    def formulate_questions(self, uncertain_detections):
        """Formulate questions about uncertain detections."""
        questions = []
        
        for uncertain in uncertain_detections:
            detection = uncertain['detection']
            element_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 0)
            
            question_text = f"Is this a {element_type}? (confidence: {confidence:.2f})"
                
            questions.append({
                'question_id': uncertain['question_id'],
                'question_text': question_text,
                'element_type': element_type,
                'image_path': uncertain['image_path'],
                'options': ['Yes', 'No', 'It\'s something else']
            })
            
        return questions
    
    def process_feedback(self, question_id, feedback, alternative_type=None):
        """Process user feedback for an uncertain detection."""
        # Find the question
        question = None
        for q in self.questions_asked:
            if q['question_id'] == question_id:
                question = q
                break
                
        if not question:
            return
            
        # Get the original detection
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
        question['status'] = 'answered'
        
        # Update uncertainty threshold based on feedback
        if feedback == 'Yes':
            # Lower threshold slightly for this element type
            self.uncertainty_thresholds[element_type] = max(0.5, 
                self.uncertainty_thresholds.get(element_type, 0.7) - 0.02)
        elif feedback == 'No':
            # Increase threshold for this element type
            self.uncertainty_thresholds[element_type] = min(0.9, 
                self.uncertainty_thresholds.get(element_type, 0.7) + 0.05)
        
        logger.info(f"Processed feedback for question {question_id}: {feedback}")
    
    def extract_learning_insights(self):
        """Extract insights from the active learning process."""
        if not self.feedback_received:
            return ["No active learning feedback data available yet."]
            
        total_questions = len(self.questions_asked)
        answered_questions = sum(1 for q in self.questions_asked if q.get('status') == 'answered')
        correct_answers = sum(1 for f in self.feedback_received if f.get('feedback') == 'Yes')
        
        insights = [
            f"Active learning has asked {total_questions} questions, with a {answered_questions/total_questions*100:.1f}% response rate.",
            f"Overall detection precision: {correct_answers/answered_questions*100:.1f}% if answered",
            f"Current uncertainty thresholds: {', '.join([f'{k}: {v:.2f}' for k,v in self.uncertainty_thresholds.items()])}"
        ]
        
        return insights
    
    def apply_learning_to_detector(self, detector):
        """Apply learning from user feedback to improve a detector."""
        if not hasattr(detector, 'supports_incremental_learning') or not detector.supports_incremental_learning():
            logger.warning(f"Detector {detector.name if hasattr(detector, 'name') else 'Unknown'} doesn't support incremental learning")
            return
            
        correct_count = 0
        incorrect_count = 0
        
        for feedback in self.feedback_received:
            question_id = feedback['question_id']
            feedback_value = feedback['feedback']
            
            # Find the question
            question = None
            for q in self.questions_asked:
                if q['question_id'] == question_id:
                    question = q
                    break
                    
            if not question:
                continue
                
            # Get the detection
            detection = question['detection']
            element_type = detection.get('type', 'unknown')
            
            if feedback_value == 'Yes':
                # Correct detection - add as positive example
                if hasattr(detector, 'add_training_example'):
                    detector.add_training_example(np.zeros((10, 10, 3)), element_type, True)  # Mock image
                elif hasattr(detector, 'add_correct_detection'):
                    detector.add_correct_detection(np.zeros((10, 10, 3)), detection)  # Mock image
                correct_count += 1
            else:
                # Incorrect detection - add as negative example
                if hasattr(detector, 'add_training_example'):
                    detector.add_training_example(np.zeros((10, 10, 3)), element_type, False)  # Mock image
                incorrect_count += 1
                
        logger.info(f"Applied active learning: {correct_count} correct and {incorrect_count} incorrect examples")

# Test image creation and manipulation
class TestImageGenerator:
    """Generates test images for UI detection testing."""
    
    def __init__(self, output_dir: str = 'test_images'):
        """Initialize the test image generator.
        
        Args:
            output_dir: Directory to save test images
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_login_form(self, image_name: str = 'login_form.png', 
                          width: int = 800, height: int = 600) -> np.ndarray:
        """Create a test image with a login form.
        
        Args:
            image_name: Filename for the output image
            width: Image width
            height: Image height
            
        Returns:
            The generated image as a numpy array
        """
        # Create blank image with white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add form container with light gray background
        cv2.rectangle(image, (200, 100), (600, 500), (240, 240, 240), -1)
        cv2.rectangle(image, (200, 100), (600, 500), (200, 200, 200), 2)
        
        # Add form title
        cv2.putText(image, "User Login", (320, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        
        # Add username input field
        cv2.rectangle(image, (250, 200), (550, 240), (255, 255, 255), -1)
        cv2.rectangle(image, (250, 200), (550, 240), (180, 180, 180), 1)
        cv2.putText(image, "Username", (255, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Add password input field
        cv2.rectangle(image, (250, 280), (550, 320), (255, 255, 255), -1)
        cv2.rectangle(image, (250, 280), (550, 320), (180, 180, 180), 1)
        cv2.putText(image, "Password", (255, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Add remember me checkbox
        cv2.rectangle(image, (250, 350), (270, 370), (255, 255, 255), -1)
        cv2.rectangle(image, (250, 350), (270, 370), (180, 180, 180), 1)
        cv2.putText(image, "Remember me", (280, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Add login button
        cv2.rectangle(image, (250, 420), (350, 460), (66, 134, 244), -1)  # Blue button
        cv2.putText(image, "Login", (277, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add register link
        cv2.putText(image, "Register", (450, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (66, 134, 244), 1)
        
        # Save the image
        output_path = self.output_dir / image_name
        cv2.imwrite(str(output_path), image)
        logger.info(f"Created login form test image: {output_path}")
        
        return image
    
    def create_search_interface(self, image_name: str = 'search_interface.png',
                              width: int = 800, height: int = 600) -> np.ndarray:
        """Create a test image with a search interface.
        
        Args:
            image_name: Filename for the output image
            width: Image width
            height: Image height
            
        Returns:
            The generated image as a numpy array
        """
        # Create blank image with light gray background
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add header area
        cv2.rectangle(image, (0, 0), (width, 80), (50, 50, 50), -1)
        cv2.putText(image, "Search Portal", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add search bar
        cv2.rectangle(image, (200, 150), (600, 190), (255, 255, 255), -1)
        cv2.rectangle(image, (200, 150), (600, 190), (180, 180, 180), 1)
        
        # Add search icon (magnifying glass)
        center = (580, 170)
        radius = 10
        handle_start = (center[0] + radius * 0.7, center[1] + radius * 0.7)
        handle_end = (center[0] + radius * 1.5, center[1] + radius * 1.5)
        
        cv2.circle(image, center, radius, (100, 100, 100), 2)
        cv2.line(image, 
                (int(handle_start[0]), int(handle_start[1])), 
                (int(handle_end[0]), int(handle_end[1])), 
                (100, 100, 100), 2)
        
        # Add filter buttons
        cv2.rectangle(image, (200, 220), (280, 250), (220, 220, 220), -1)
        cv2.rectangle(image, (200, 220), (280, 250), (180, 180, 180), 1)
        cv2.putText(image, "Filter", (215, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        cv2.rectangle(image, (300, 220), (400, 250), (220, 220, 220), -1)
        cv2.rectangle(image, (300, 220), (400, 250), (180, 180, 180), 1)
        cv2.putText(image, "Categories", (315, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        # Add result items (simplified)
        for i in range(3):
            y_pos = 300 + i * 80
            cv2.rectangle(image, (200, y_pos), (600, y_pos + 60), (255, 255, 255), -1)
            cv2.rectangle(image, (200, y_pos), (600, y_pos + 60), (200, 200, 200), 1)
            cv2.putText(image, f"Result Item {i+1}", (220, y_pos + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        
        # Save the image
        output_path = self.output_dir / image_name
        cv2.imwrite(str(output_path), image)
        logger.info(f"Created search interface test image: {output_path}")
        
        return image
    
    def create_dashboard(self, image_name: str = 'dashboard.png',
                        width: int = 800, height: int = 600) -> np.ndarray:
        """Create a test image with a dashboard interface.
        
        Args:
            image_name: Filename for the output image
            width: Image width
            height: Image height
            
        Returns:
            The generated image as a numpy array
        """
        # Create blank image with white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add header area
        cv2.rectangle(image, (0, 0), (width, 60), (41, 128, 185), -1)
        cv2.putText(image, "Dashboard", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add sidebar
        cv2.rectangle(image, (0, 60), (180, height), (240, 240, 240), -1)
        menu_items = ["Home", "Analytics", "Reports", "Settings", "Logout"]
        for i, item in enumerate(menu_items):
            y_pos = 100 + i * 50
            cv2.putText(image, item, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        
        # Add content area with cards
        for i in range(2):
            for j in range(2):
                x_pos = 210 + j * 280
                y_pos = 90 + i * 240
                
                # Card background
                cv2.rectangle(image, (x_pos, y_pos), (x_pos + 250, y_pos + 200), (250, 250, 250), -1)
                cv2.rectangle(image, (x_pos, y_pos), (x_pos + 250, y_pos + 200), (220, 220, 220), 1)
                
                # Card title
                cv2.putText(image, f"Card {i*2+j+1}", (x_pos + 20, y_pos + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
                
                # Card content
                cv2.rectangle(image, (x_pos + 20, y_pos + 60), (x_pos + 230, y_pos + 120), (245, 245, 245), -1)
                
                # Card button
                btn_color = [(46, 204, 113), (231, 76, 60), (52, 152, 219), (155, 89, 182)][i*2+j]
                cv2.rectangle(image, (x_pos + 20, y_pos + 150), (x_pos + 120, y_pos + 180), btn_color, -1)
                cv2.putText(image, "Details", (x_pos + 35, y_pos + 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the image
        output_path = self.output_dir / image_name
        cv2.imwrite(str(output_path), image)
        logger.info(f"Created dashboard test image: {output_path}")
        
        return image

# Implement a basic OpenCV detector for testing
class BasicOpenCVDetector:
    """Detects UI elements using OpenCV template matching."""
    
    def __init__(self, template_dir: str = 'ui_templates'):
        """Initialize the OpenCV detector.
        
        Args:
            template_dir: Directory containing template images
        """
        self.name = "BasicOpenCVDetector"
        self.description = "Detects UI elements using OpenCV template matching"
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.load_templates()
        
    def load_templates(self) -> None:
        """Load template images from the template directory."""
        self.templates = {}
        template_count = 0
        
        # Load templates by category
        for category in ['buttons', 'inputs', 'checkboxes', 'icons', 'toggles']:
            category_dir = self.template_dir / category
            if not category_dir.exists():
                continue
                
            self.templates[category] = []
            
            for template_file in category_dir.glob("*.png"):
                template = cv2.imread(str(template_file))
                if template is not None:
                    self.templates[category].append({
                        'image': template,
                        'name': template_file.stem,
                        'path': str(template_file)
                    })
                    template_count += 1
        
        # Also check for learned templates
        learned_dir = self.template_dir / 'learned'
        if learned_dir.exists():
            for category in ['button', 'text_input', 'checkbox', 'icon', 'dropdown', 'toggle']:
                category_dir = learned_dir / category
                if not category_dir.exists():
                    continue
                    
                if category not in self.templates:
                    self.templates[category] = []
                    
                for template_file in category_dir.glob("*.png"):
                    template = cv2.imread(str(template_file))
                    if template is not None:
                        self.templates[category].append({
                            'image': template,
                            'name': template_file.stem,
                            'path': str(template_file),
                            'learned': True
                        })
                        template_count += 1
        
        logger.info(f"Loaded {template_count} templates for UI detection")
    
    def detect_elements(self, image: np.ndarray, context: Dict = None) -> List[Dict]:
        """Detect UI elements in the image using template matching.
        
        Args:
            image: Image as numpy array
            context: Optional context information
            
        Returns:
            List of detected UI elements
        """
        if not self.templates:
            logger.warning("No templates available for detection")
            return []
            
        detected_elements = []
        
        # Determine which categories to focus on based on context
        categories_to_check = list(self.templates.keys())
        if context and 'target_element_type' in context:
            target_type = context['target_element_type']
            # Map target type to template categories
            type_to_category = {
                'button': ['buttons'],
                'text_input': ['inputs'],
                'checkbox': ['checkboxes'],
                'icon': ['icons'],
                'toggle': ['toggles']
            }
            categories_to_check = type_to_category.get(target_type, categories_to_check)
        
        # Check for each category of templates
        for category, templates in self.templates.items():
            if category not in categories_to_check:
                continue
                
            for template_info in templates:
                template = template_info['image']
                
                # Skip templates larger than the image
                if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                    continue
                    
                # Perform template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.7  # Adjust threshold as needed
                loc = np.where(result >= threshold)
                
                for pt in zip(*loc[::-1]):
                    # Check for overlapping detections
                    overlap = False
                    for elem in detected_elements:
                        rect = elem['rect']
                        if self._check_overlap(
                            (pt[0], pt[1], template.shape[1], template.shape[0]),
                            (rect['x'], rect['y'], rect['width'], rect['height'])
                        ):
                            overlap = True
                            break
                    
                    if overlap:
                        continue
                        
                    # Map category to element type
                    category_to_type = {
                        'buttons': 'button',
                        'inputs': 'text_input',
                        'checkboxes': 'checkbox',
                        'icons': 'icon',
                        'toggles': 'toggle',
                        'button': 'button',
                        'text_input': 'text_input',
                        'checkbox': 'checkbox',
                        'icon': 'icon',
                        'toggle': 'toggle'
                    }
                    
                    element_type = category_to_type.get(category, 'unknown')
                    confidence = float(result[pt[1], pt[0]])
                    
                    # Create element data
                    element = {
                        'id': f"opencv_{element_type}_{len(detected_elements)}",
                        'type': element_type,
                        'rect': {
                            'x': int(pt[0]),
                            'y': int(pt[1]),
                            'width': template.shape[1],
                            'height': template.shape[0]
                        },
                        'confidence': confidence,
                        'text': '',  # OpenCV doesn't extract text
                        'source_detector': self.name,
                        'template_name': template_info['name']
                    }
                    
                    detected_elements.append(element)
        
        return detected_elements
    
    def _check_overlap(self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Check if one rectangle is to the left of the other
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
            
        # Check if one rectangle is above the other
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
            
        return True
    
    def supports_incremental_learning(self) -> bool:
        """Whether this detector supports incremental learning."""
        return True
    
    def provide_feedback(self, detection_results: List[Dict], correct_results: List[Dict]) -> None:
        """Process feedback to improve detection in future iterations."""
        # For BasicOpenCVDetector, we simply reload templates to pick up any new ones
        self.load_templates()


def run_adaptive_ui_detection_test():
    """Run tests demonstrating adaptive UI detection capabilities."""
    logger.info("Starting Adaptive UI Detection Test")
    
    # Create test directories
    templates_dir = Path('ui_templates')
    os.makedirs(templates_dir, exist_ok=True)
    for category in ['buttons', 'inputs', 'checkboxes', 'icons', 'toggles', 'learned']:
        os.makedirs(templates_dir / category, exist_ok=True)
    
    # Create test images
    image_generator = TestImageGenerator(output_dir='test_images')
    login_image = image_generator.create_login_form()
    search_image = image_generator.create_search_interface()
    dashboard_image = image_generator.create_dashboard()
    
    # Create the detector registry
    registry = EnhancedDetectorRegistry()
    
    # Create and initialize detectors
    basic_detector = BasicOpenCVDetector(template_dir='ui_templates')
    
    # Create a config path for the multi-API detector
    config_path = 'api_config.json'
    api_config = {
        'api_preferences': ['gemini', 'huggingface', 'openai', 'groq'],
        'gemini': {
            'api_key': 'AIzaSyCo6WNPcAbv852AYM-PMyUGbDxqgoXnmiQ',
            'model': 'gemini-pro-vision'
        },
        'huggingface': {
            'api_key': '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',
            'model': 'Salesforce/blip-image-captioning-large'
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(api_config, f, indent=2)
    
    # Initialize the multi-API vision detector
    multi_api_detector = MultiAPIVisionDetector(config_path=config_path)
    
    # Initialize neural network detector
    neural_detector = MockNeuralNetworkDetector()
    
    # Initialize the self-expanding template library
    template_library = SelfExpandingTemplateLibrary('ui_templates')
    
    # Initialize the adaptive detector factory
    detector_factory = AdaptiveDetectorFactory(registry=registry, memory_path='adaptive_detectors')
    
    # Initialize the cross-context learner
    cross_context_learner = MockCrossContextLearner(memory_path='cross_context_knowledge')
    
    # Initialize the active learning interface
    active_learning_interface = MockActiveLearningInterface(memory_path='active_learning_memory')
    
    # Register detectors
    registry.register_detector(basic_detector)
    registry.register_detector(multi_api_detector)
    registry.register_detector(neural_detector)
    
    # Initialize the detection coordinator
    coordinator = MultimodalDetectionCoordinator(detector_registry=registry)
    
    # Step 1: Run initial detections using basic detector
    logger.info("Step 1: Initial detections with basic detector")
    
    login_elements = registry.detect_with_detector(
        'BasicOpenCVDetector',
        login_image,
        {'description': 'Login form'}
    )
    
    logger.info(f"Detected {len(login_elements)} elements in login form using basic detector")
    
    # Step 2: Add simulated human feedback (correct elements)
    logger.info("Step 2: Adding simulated feedback")
    
    # Simulated correct results for login form
    correct_login_elements = [
        {
            'type': 'text_input',
            'rect': {'x': 250, 'y': 200, 'width': 300, 'height': 40},
            'text': 'Username',
            'confidence': 1.0
        },
        {
            'type': 'text_input',
            'rect': {'x': 250, 'y': 280, 'width': 300, 'height': 40},
            'text': 'Password',
            'confidence': 1.0
        },
        {
            'type': 'checkbox',
            'rect': {'x': 250, 'y': 350, 'width': 20, 'height': 20},
            'text': 'Remember me',
            'confidence': 1.0
        },
        {
            'type': 'button',
            'rect': {'x': 250, 'y': 420, 'width': 100, 'height': 40},
            'text': 'Login',
            'confidence': 1.0
        }
    ]
    
    # Extract image patches for template learning
    for element in correct_login_elements:
        x, y = element['rect']['x'], element['rect']['y']
        w, h = element['rect']['width'], element['rect']['height']
        element_image = login_image[y:y+h, x:x+w]
        
        # Add element image to template library for learning
        template_library.add_template(
            element_image,
            element['type'],
            element.get('text', ''),
            {'form': 'login'},
            confidence=1.0
        )
    
    # Step 3: Test self-expanding template library
    logger.info("Step 3: Testing self-expanding template library")
    
    # Reload templates to include newly added ones
    basic_detector.load_templates()
    
    # Run detection again with basic detector
    login_elements_after = registry.detect_with_detector(
        'BasicOpenCVDetector',
        login_image,
        {'description': 'Login form'}
    )
    
    logger.info(f"Detected {len(login_elements_after)} elements after template learning")
    
    # Step 4: Test specialized detector creation
    logger.info("Step 4: Creating specialized detectors")
    
    # Create specialized detector for buttons
    button_detector_id = detector_factory.create_specialized_detector(
        'BasicOpenCVDetector',
        'element_type',
        element_type='button'
    )
    
    logger.info(f"Created specialized button detector: {button_detector_id}")
    
    # Create specialized detector for login forms
    login_detector_id = detector_factory.create_specialized_detector(
        'BasicOpenCVDetector',
        'context',
        context='login'
    )
    
    logger.info(f"Created specialized login form detector: {login_detector_id}")
    
    # Step 5: Test multi-API vision detection
    logger.info("Step 5: Testing multi-API vision detection")
    
    # Only run this if API keys are configured
    if multi_api_detector.active_api:
        dashboard_elements = registry.detect_with_detector(
            'MultiAPIVisionDetector',
            dashboard_image,
            {'description': 'Dashboard with navigation menu and cards'}
        )
        
        logger.info(f"Detected {len(dashboard_elements)} elements in dashboard using {multi_api_detector.active_api}")
    else:
        logger.warning("Skipping multi-API vision detection test (no API keys configured)")
    
    # Step 6: Test end-to-end adaptive detection with all components
    logger.info("Step 6: Testing end-to-end adaptive detection")
    
    # Use the coordinator for full detection
    search_results = coordinator.detect_elements(
        search_image,
        context={'description': 'Search interface with search bar and filters'}
    )
    
    logger.info(f"Detected {len(search_results)} elements using full detection pipeline")
    
    # Step 7: Test cross-context learning
    logger.info("Step 7: Testing cross-context learning")
    
    # Add knowledge from login form context
    login_context = 'login_form'
    for element in correct_login_elements:
        # Add the element knowledge to the learner
        cross_context_learner.add_context_knowledge(
            login_context,
            element,
            login_image
        )
    
    # Add knowledge from search interface context
    search_context = 'search_interface'
    search_elements = [
        {
            'type': 'text_input',
            'rect': {'x': 200, 'y': 150, 'width': 400, 'height': 40},
            'text': 'Search',
            'confidence': 1.0
        },
        {
            'type': 'button',
            'rect': {'x': 200, 'y': 220, 'width': 80, 'height': 30},
            'text': 'Filter',
            'confidence': 1.0
        },
        {
            'type': 'button',
            'rect': {'x': 300, 'y': 220, 'width': 100, 'height': 30},
            'text': 'Categories',
            'confidence': 1.0
        }
    ]
    
    for element in search_elements:
        # Add the element knowledge to the learner
        cross_context_learner.add_context_knowledge(
            search_context,
            element,
            search_image
        )
    
    # Test knowledge transfer between contexts
    transfer_result = cross_context_learner.transfer_knowledge(
        login_context,
        'dashboard',
        element_type='button'
    )
    
    logger.info(f"Cross-context transfer result: {transfer_result['success']}")
    
    # Find similar elements across contexts
    similar_buttons = cross_context_learner.find_similar_elements(
        {'type': 'button', 'text': 'Login'},
        context_name=search_context
    )
    
    logger.info(f"Found {len(similar_buttons)} similar button elements across contexts")
    
    # Get context knowledge stats
    context_stats = cross_context_learner.get_context_knowledge_stats()
    logger.info(f"Cross-context knowledge stats: {context_stats['context_count']} contexts, {context_stats['element_type_count']} element types")
    
    # Display statistics
    detector_stats = detector_factory.get_detector_creation_stats()
    logger.info(f"Adaptive detector statistics: {detector_stats}")
    
    template_stats = template_library.get_statistics()
    logger.info(f"Template library statistics: {template_stats}")
    
    # Test active learning interface
    logger.info("Step 8: Testing active learning interface")
    
    # Process detection results to find uncertain detections
    uncertain_detections = active_learning_interface.process_detection_results(
        registry.detect_with_detector('BasicOpenCVDetector', search_image),
        search_image,
        max_questions=2
    )
    
    # Formulate questions about uncertain detections
    questions = active_learning_interface.formulate_questions(uncertain_detections)
    
    # Simulate user feedback for active learning
    if questions:
        for question in questions:
            # Simulate 'Yes' response for first question and 'No' for others
            feedback = 'Yes' if question == questions[0] else 'No'
            active_learning_interface.process_feedback(question['question_id'], feedback)
        
        # Get learning insights
        insights = active_learning_interface.extract_learning_insights()
        logger.info(f"Active learning insights: {insights}")
    else:
        logger.info("No active learning questions were generated")
    
    # Apply learning to improve the neural detector
    active_learning_interface.apply_learning_to_detector(neural_detector)
    
    logger.info("Adaptive UI Detection Test completed")


if __name__ == "__main__":
    run_adaptive_ui_detection_test()
