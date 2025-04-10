#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Detector Factory for UI detection.
This module dynamically creates specialized detectors based on successful detection patterns.
"""

import os
import json
import uuid
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

class AdaptiveDetectorFactory:
    """
    Factory that creates specialized detectors based on learning patterns from successful detections.
    This implements a core component of the system's ability to learn and improve itself over time.
    """
    
    def __init__(self, registry=None, memory_path: str = None):
        """Initialize the adaptive detector factory.
        
        Args:
            registry: The detector registry to register new detectors with
            memory_path: Path for storing detector specifications
        """
        self.registry = registry
        self.memory_path = Path(memory_path) if memory_path else Path('adaptive_detectors')
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Load existing specialized detectors
        self.specialized_detectors = self._load_specialized_detectors()
        
        # Track performance metrics for adaptive learning
        self.performance_history = {}
        
        # Track creation statistics
        self.detectors_created = 0
        self.detector_enhancements = 0
        
        logger.info(f"Initialized AdaptiveDetectorFactory with {len(self.specialized_detectors)} existing specialized detectors")
    
    def _load_specialized_detectors(self) -> Dict:
        """Load existing specialized detector specifications."""
        specs_file = self.memory_path / 'specialized_detectors.json'
        if os.path.exists(specs_file):
            try:
                with open(specs_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading specialized detectors: {e}")
                return {}
        return {}
    
    def _save_specialized_detectors(self) -> None:
        """Save specialized detector specifications."""
        specs_file = self.memory_path / 'specialized_detectors.json'
        try:
            with open(specs_file, 'w') as f:
                json.dump(self.specialized_detectors, f, indent=2)
            logger.info(f"Saved {len(self.specialized_detectors)} specialized detectors")
        except IOError as e:
            logger.error(f"Error saving specialized detectors: {e}")
    
    def analyze_detection_patterns(self, detection_history: List[Dict], 
                                   feedback_history: List[Dict]) -> Dict:
        """Analyze detection patterns to identify opportunities for specialization.
        
        Args:
            detection_history: History of detection results
            feedback_history: History of feedback on detections
            
        Returns:
            Dictionary of potential specialization opportunities
        """
        # Map detectors to their successful element types
        detector_element_map = {}
        detector_context_map = {}
        
        # Analyze detection patterns from feedback history
        for feedback in feedback_history:
            detector_name = feedback.get('detector', '')
            element_type = feedback.get('element_type', '')
            context = feedback.get('context', '')
            success = feedback.get('success', False)
            confidence = feedback.get('confidence', 0.0)
            
            if not detector_name or not element_type:
                continue
                
            # Track successful detections by detector and element type
            if success:
                if detector_name not in detector_element_map:
                    detector_element_map[detector_name] = {}
                
                if element_type not in detector_element_map[detector_name]:
                    detector_element_map[detector_name][element_type] = {
                        'count': 0,
                        'total_confidence': 0.0
                    }
                
                detector_element_map[detector_name][element_type]['count'] += 1
                detector_element_map[detector_name][element_type]['total_confidence'] += confidence
                
                # Track context associations
                if context:
                    if detector_name not in detector_context_map:
                        detector_context_map[detector_name] = {}
                    
                    if context not in detector_context_map[detector_name]:
                        detector_context_map[detector_name][context] = {
                            'count': 0,
                            'total_confidence': 0.0,
                            'element_types': {}
                        }
                    
                    detector_context_map[detector_name][context]['count'] += 1
                    detector_context_map[detector_name][context]['total_confidence'] += confidence
                    
                    # Track element types within this context
                    if element_type not in detector_context_map[detector_name][context]['element_types']:
                        detector_context_map[detector_name][context]['element_types'][element_type] = 0
                    
                    detector_context_map[detector_name][context]['element_types'][element_type] += 1
        
        # Find specialization opportunities
        specialization_opportunities = {
            'element_type': [],
            'context': [],
            'combined': []
        }
        
        # Check for element type specialization opportunities
        for detector_name, element_types in detector_element_map.items():
            for element_type, stats in element_types.items():
                # Only consider strong patterns (multiple successful detections with good confidence)
                if stats['count'] >= 3 and stats['total_confidence'] / stats['count'] >= 0.8:
                    specialization_opportunities['element_type'].append({
                        'detector': detector_name,
                        'element_type': element_type,
                        'success_count': stats['count'],
                        'avg_confidence': stats['total_confidence'] / stats['count']
                    })
        
        # Check for context specialization opportunities
        for detector_name, contexts in detector_context_map.items():
            for context, stats in contexts.items():
                # Only consider strong patterns
                if stats['count'] >= 3 and stats['total_confidence'] / stats['count'] >= 0.8:
                    specialization_opportunities['context'].append({
                        'detector': detector_name,
                        'context': context,
                        'success_count': stats['count'],
                        'avg_confidence': stats['total_confidence'] / stats['count'],
                        'element_types': stats['element_types']
                    })
        
        # Check for combined specialization opportunities (element type + context)
        for detector_name, contexts in detector_context_map.items():
            for context, context_stats in contexts.items():
                for element_type, count in context_stats['element_types'].items():
                    # Only consider strong patterns
                    if count >= 2 and context_stats['count'] >= 3:
                        specialization_opportunities['combined'].append({
                            'detector': detector_name,
                            'context': context,
                            'element_type': element_type,
                            'success_count': count,
                            'context_count': context_stats['count'],
                            'avg_confidence': context_stats['total_confidence'] / context_stats['count']
                        })
        
        return specialization_opportunities
    
    def create_specialized_detector(self, base_detector_name: str, specialization_type: str,
                                   element_type: str = None, context: str = None) -> str:
        """Create a specialized detector based on a base detector.
        
        Args:
            base_detector_name: Name of the base detector to specialize
            specialization_type: Type of specialization ('element_type', 'context', or 'combined')
            element_type: Element type to specialize for (if applicable)
            context: Context to specialize for (if applicable)
            
        Returns:
            ID of the created specialized detector
        """
        if not self.registry:
            logger.warning("Cannot create specialized detector without registry")
            return None
            
        # Generate a unique ID for the detector
        detector_id = f"specialized_{specialization_type}_{str(uuid.uuid4())[:8]}"
        
        # Create detector specification
        spec = {
            'id': detector_id,
            'base_detector': base_detector_name,
            'specialization_type': specialization_type,
            'element_type': element_type,
            'context': context,
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'performance': {
                'calls': 0,
                'successes': 0,
                'total_confidence': 0.0
            }
        }
        
        # Create a user-friendly name
        if specialization_type == 'element_type' and element_type:
            name = f"{element_type.capitalize()}Specialized{base_detector_name}"
        elif specialization_type == 'context' and context:
            name = f"{context.capitalize()}Specialized{base_detector_name}"
        elif specialization_type == 'combined' and element_type and context:
            name = f"{context.capitalize()}{element_type.capitalize()}Specialized{base_detector_name}"
        else:
            name = f"Specialized{base_detector_name}_{detector_id}"
        
        spec['name'] = name
        
        # Add to specialized detectors dictionary
        self.specialized_detectors[detector_id] = spec
        self._save_specialized_detectors()
        
        # Create and register the specialized detector
        if self.registry:
            try:
                # Import the detector dynamically based on name
                base_detector = self.registry.get_detector(base_detector_name)
                if not base_detector:
                    logger.warning(f"Base detector {base_detector_name} not found")
                    return None
                
                # Create the specialized detector
                specialized_detector = self._create_specialized_detector_instance(base_detector, spec)
                
                # Register with the registry
                self.registry.register_detector(specialized_detector, detector_id, name)
                
                self.detectors_created += 1
                logger.info(f"Created and registered specialized detector: {name}")
                
                return detector_id
            except Exception as e:
                logger.error(f"Error creating specialized detector: {e}")
                return None
        else:
            return detector_id
    
    def _create_specialized_detector_instance(self, base_detector: Any, spec: Dict) -> Any:
        """Create a specialized detector instance based on a specification."""
        # Create a wrapper class for the specialized detector
        class SpecializedDetector:
            def __init__(self, base, specification):
                self.base = base
                self.spec = specification
                self.name = specification['name']
                self.description = f"Specialized detector for {specification['specialization_type']}"
                self.element_type = specification.get('element_type')
                self.context = specification.get('context')
                self.calls = 0
                self.successes = 0
                
            def detect_elements(self, image, context=None):
                """Detect elements using specialized parameters."""
                self.calls += 1
                
                # Check if this detector is applicable based on its specialization
                applicable = True
                if self.spec['specialization_type'] == 'context' and self.context:
                    applicable = context and self.context.lower() in context.get('description', '').lower()
                elif self.spec['specialization_type'] == 'combined' and self.element_type and self.context:
                    applicable = context and self.context.lower() in context.get('description', '').lower()
                
                if not applicable:
                    return []
                
                # Enhance detection parameters based on specialization
                enhanced_context = context.copy() if context else {}
                
                if self.spec['specialization_type'] == 'element_type' and self.element_type:
                    enhanced_context['target_element_type'] = self.element_type
                
                # Perform detection with the base detector
                results = self.base.detect_elements(image, enhanced_context)
                
                # Filter results if specializing for element type
                if self.spec['specialization_type'] in ['element_type', 'combined'] and self.element_type:
                    results = [elem for elem in results if elem.get('type') == self.element_type]
                
                # Enhance confidence for specialized detections
                for elem in results:
                    # Slightly boost confidence for specialized detections
                    elem['confidence'] = min(1.0, elem.get('confidence', 0.8) * 1.05)
                    elem['source_detector'] = self.name
                
                return results
                
            def supports_incremental_learning(self):
                """Whether this detector supports incremental learning."""
                return hasattr(self.base, 'supports_incremental_learning') and self.base.supports_incremental_learning()
                
            def provide_feedback(self, detection_results, correct_results):
                """Provide feedback to improve detection."""
                if hasattr(self.base, 'provide_feedback'):
                    self.base.provide_feedback(detection_results, correct_results)
                    
                # Update success statistics
                if detection_results:
                    self.successes += 1
        
        # Create and return an instance of the specialized detector
        return SpecializedDetector(base_detector, spec)
    
    def update_detector_performance(self, detector_id: str, success: bool, confidence: float) -> None:
        """Update performance metrics for a specialized detector.
        
        Args:
            detector_id: ID of the specialized detector
            success: Whether the detection was successful
            confidence: Confidence score of the detection
        """
        if detector_id in self.specialized_detectors:
            spec = self.specialized_detectors[detector_id]
            spec['performance']['calls'] += 1
            
            if success:
                spec['performance']['successes'] += 1
                spec['performance']['total_confidence'] += confidence
            
            self._save_specialized_detectors()
    
    def get_detector_creation_stats(self) -> Dict:
        """Get statistics about detector creation.
        
        Returns:
            Dictionary with detector creation statistics
        """
        return {
            'detectors_created': self.detectors_created,
            'detector_enhancements': self.detector_enhancements,
            'specialized_detectors': len(self.specialized_detectors),
            'specialization_types': {
                'element_type': len([d for d in self.specialized_detectors.values() if d['specialization_type'] == 'element_type']),
                'context': len([d for d in self.specialized_detectors.values() if d['specialization_type'] == 'context']),
                'combined': len([d for d in self.specialized_detectors.values() if d['specialization_type'] == 'combined'])
            }
        }

def create_adaptive_detector_factory(registry=None, memory_path=None):
    """Create and initialize the adaptive detector factory.
    
    Args:
        registry: The detector registry to register new detectors with
        memory_path: Path for storing detector specifications
        
    Returns:
        Initialized AdaptiveDetectorFactory instance
    """
    return AdaptiveDetectorFactory(registry, memory_path)
