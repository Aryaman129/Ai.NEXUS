"""
Multimodal Context Awareness module for NEXUS UI Detection.

This module combines visual detection with system state awareness to improve UI element detection.
It tracks system state, active applications, and user interaction patterns to prioritize detection
of relevant UI elements based on context.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Setup logging
logger = logging.getLogger("adaptive_ui_detection")

class MultimodalContextAwareness:
    """
    Combines visual detection with system state monitoring to understand 
    when certain UI elements are more likely to be relevant.
    """
    
    def __init__(self, memory_path: str = None):
        self.name = "MultimodalContextAwareness"
        self.memory_path = memory_path or "multimodal_context_memory"
        
        # Create memory directory if it doesn't exist
        if self.memory_path:
            os.makedirs(self.memory_path, exist_ok=True)
        
        # Initialize context state tracking
        self.active_application = None
        self.active_window_title = None
        self.previous_interactions = []
        self.previous_detections = []
        self.system_state = {}
        
        # Context prioritization scores
        self.element_priorities = defaultdict(lambda: defaultdict(float))
        self.application_patterns = defaultdict(dict)
        self.workflow_transitions = defaultdict(lambda: defaultdict(int))
        
        # Load existing context memory if available
        self.load_context_memory()
        
        logger.info(f"Initialized multimodal context awareness module")
    
    def load_context_memory(self):
        """Load context memory from disk if available."""
        if not self.memory_path:
            return
            
        memory_file = os.path.join(self.memory_path, "context_memory.json")
        
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                    
                # Load element priorities
                if 'element_priorities' in memory_data:
                    self.element_priorities = defaultdict(lambda: defaultdict(float))
                    for app, elements in memory_data['element_priorities'].items():
                        for element_type, priority in elements.items():
                            self.element_priorities[app][element_type] = priority
                
                # Load application patterns
                if 'application_patterns' in memory_data:
                    self.application_patterns = defaultdict(dict)
                    for app, patterns in memory_data['application_patterns'].items():
                        self.application_patterns[app] = patterns
                
                # Load workflow transitions
                if 'workflow_transitions' in memory_data:
                    self.workflow_transitions = defaultdict(lambda: defaultdict(int))
                    for app, transitions in memory_data['workflow_transitions'].items():
                        for from_state, to_states in transitions.items():
                            for to_state, count in to_states.items():
                                self.workflow_transitions[app][from_state] = defaultdict(int)
                                self.workflow_transitions[app][from_state][to_state] = count
                
                logger.info(f"Loaded multimodal context memory from {memory_file}")
            except Exception as e:
                logger.error(f"Error loading context memory: {e}")
    
    def save_context_memory(self):
        """Save context memory to disk."""
        if not self.memory_path:
            return
            
        memory_file = os.path.join(self.memory_path, "context_memory.json")
        
        try:
            # Convert defaultdict to regular dict for JSON serialization
            memory_data = {
                'element_priorities': {
                    app: dict(elements) for app, elements in self.element_priorities.items()
                },
                'application_patterns': dict(self.application_patterns),
                'workflow_transitions': {
                    app: {
                        from_state: dict(to_states) for from_state, to_states in transitions.items()
                    } for app, transitions in self.workflow_transitions.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            logger.info(f"Saved multimodal context memory to {memory_file}")
        except Exception as e:
            logger.error(f"Error saving context memory: {e}")
    
    def update_system_state(self, system_info: Dict[str, Any]):
        """Update the current system state information."""
        self.system_state = system_info
        
        # Extract relevant state information
        self.active_application = system_info.get('active_application')
        self.active_window_title = system_info.get('window_title')
        
        logger.info(f"Updated system state: {self.active_application} - {self.active_window_title}")
    
    def record_interaction(self, element_type: str, element_data: Dict[str, Any]):
        """Record a user interaction with a UI element."""
        if not self.active_application:
            return
            
        # Create interaction record
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'application': self.active_application,
            'window_title': self.active_window_title,
            'element_type': element_type,
            'element_data': element_data
        }
        
        # Add to interaction history
        self.previous_interactions.append(interaction)
        
        # Keep only last 20 interactions
        if len(self.previous_interactions) > 20:
            self.previous_interactions = self.previous_interactions[-20:]
        
        # Update element priority for this application
        self.element_priorities[self.active_application][element_type] += 0.1
        
        # Update workflow transitions if there was a previous interaction
        if len(self.previous_interactions) > 1:
            prev = self.previous_interactions[-2]
            from_state = prev['element_type']
            to_state = element_type
            
            self.workflow_transitions[self.active_application][from_state][to_state] += 1
        
        logger.info(f"Recorded interaction with {element_type} in {self.active_application}")
        
        # Save updated context memory
        self.save_context_memory()
    
    def record_detections(self, detections: List[Dict[str, Any]]):
        """Record UI element detections for context learning."""
        if not self.active_application or not detections:
            return
            
        # Update application patterns based on detected elements
        element_types = [d['type'] for d in detections if 'type' in d]
        
        # Count element types
        type_counts = {}
        for element_type in element_types:
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        # Update application patterns
        self.application_patterns[self.active_application]['element_type_distribution'] = type_counts
        
        # Store the detections with context
        self.previous_detections = [{
            'timestamp': datetime.now().isoformat(),
            'application': self.active_application,
            'window_title': self.active_window_title,
            **detection
        } for detection in detections]
        
        logger.info(f"Recorded {len(detections)} detections in {self.active_application}")
    
    def enhance_detection_confidence(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance detection confidence based on context awareness."""
        if not self.active_application or not detections:
            return detections
            
        enhanced_detections = []
        
        for detection in detections:
            element_type = detection.get('type')
            if not element_type:
                enhanced_detections.append(detection)
                continue
                
            # Get base confidence
            confidence = detection.get('confidence', 0.7)
            
            # Apply context-based confidence adjustment
            context_boost = 0.0
            
            # 1. Boost based on element priority in this application
            priority_boost = self.element_priorities[self.active_application][element_type] * 0.05
            context_boost += min(priority_boost, 0.15)  # Cap at 15% boost
            
            # 2. Boost based on workflow transitions
            if self.previous_interactions and self.workflow_transitions[self.active_application]:
                prev_type = self.previous_interactions[-1]['element_type']
                transition_count = self.workflow_transitions[self.active_application][prev_type][element_type]
                
                if transition_count > 0:
                    transition_boost = min(transition_count * 0.02, 0.1)  # Cap at 10% boost
                    context_boost += transition_boost
            
            # 3. Adjust based on element type distribution in this application
            type_distribution = self.application_patterns[self.active_application].get('element_type_distribution', {})
            if element_type in type_distribution and sum(type_distribution.values()) > 0:
                type_ratio = type_distribution[element_type] / sum(type_distribution.values())
                distribution_boost = type_ratio * 0.05  # Max 5% boost
                context_boost += distribution_boost
            
            # Apply the context boost, capped at 25% total boost
            context_boost = min(context_boost, 0.25)
            enhanced_confidence = min(confidence + context_boost, 0.99)
            
            # Create enhanced detection
            enhanced_detection = detection.copy()
            enhanced_detection['confidence'] = enhanced_confidence
            enhanced_detection['original_confidence'] = confidence
            enhanced_detection['context_boost'] = context_boost
            
            enhanced_detections.append(enhanced_detection)
        
        logger.info(f"Enhanced detection confidence for {len(enhanced_detections)} elements")
        return enhanced_detections
    
    def predict_relevant_ui_elements(self) -> List[str]:
        """Predict which UI elements are most likely to be relevant in the current context."""
        if not self.active_application:
            return []
            
        relevant_elements = []
        
        # Sort element types by priority for this application
        app_priorities = self.element_priorities[self.active_application]
        sorted_elements = sorted(app_priorities.items(), key=lambda x: x[1], reverse=True)
        
        # Add top 5 most prioritized elements
        relevant_elements = [element_type for element_type, _ in sorted_elements[:5]]
        
        # Add elements likely to be next in the workflow
        if self.previous_interactions:
            prev_type = self.previous_interactions[-1]['element_type']
            transitions = self.workflow_transitions[self.active_application][prev_type]
            
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
            likely_next_elements = [element_type for element_type, count in sorted_transitions if count > 1]
            
            for element_type in likely_next_elements:
                if element_type not in relevant_elements:
                    relevant_elements.append(element_type)
        
        logger.info(f"Predicted relevant elements: {relevant_elements}")
        return relevant_elements
    
    def get_context_enriched_search_parameters(self) -> Dict[str, Any]:
        """Get search parameters enriched with context information."""
        search_params = {
            'active_application': self.active_application,
            'window_title': self.active_window_title,
            'relevant_elements': self.predict_relevant_ui_elements(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add workflow context if available
        if self.previous_interactions:
            search_params['recent_interactions'] = [
                i['element_type'] for i in self.previous_interactions[-3:]
            ]
        
        return search_params
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the context awareness module."""
        return {
            'name': self.name,
            'active_application': self.active_application,
            'tracked_applications': list(self.element_priorities.keys()),
            'interaction_history_size': len(self.previous_interactions),
            'workflow_transitions': {
                app: {
                    from_state: len(to_states) 
                    for from_state, to_states in transitions.items()
                }
                for app, transitions in self.workflow_transitions.items()
            },
            'timestamp': datetime.now().isoformat()
        }
