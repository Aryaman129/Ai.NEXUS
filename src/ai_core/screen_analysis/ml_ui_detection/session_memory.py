"""
Session Memory for UI Elements module for NEXUS UI Detection.

This module tracks user interactions with detected UI elements over a session,
building a memory of important elements and their relevance to the user's workflow.
It enables the system to prioritize detection of elements that are frequently used.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Setup logging
logger = logging.getLogger("adaptive_ui_detection")

class SessionMemory:
    """
    Tracks how users interact with detected elements to prioritize 
    detection of frequently used components.
    """
    
    def __init__(self, memory_path: str = None, session_expiry_hours: int = 24):
        self.name = "SessionMemory"
        self.memory_path = memory_path or "session_memory"
        self.session_expiry_hours = session_expiry_hours
        
        # Create memory directory if it doesn't exist
        if self.memory_path:
            os.makedirs(self.memory_path, exist_ok=True)
        
        # Initialize session tracking
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = datetime.now()
        
        # Interaction metrics
        self.element_interactions = defaultdict(int)  # Count of interactions with each element type
        self.element_successes = defaultdict(int)     # Count of successful interactions
        self.element_failures = defaultdict(int)      # Count of failed interactions
        self.interaction_sequence = []                # Sequence of interactions in this session
        self.location_heatmap = defaultdict(int)      # Heatmap of interaction locations
        
        # Element identification metrics
        self.detection_counts = defaultdict(int)      # Count of detections by element type
        self.detection_accuracy = defaultdict(list)   # List of detection accuracies for each element type
        
        # Session-level metrics
        self.total_interactions = 0
        self.successful_interactions = 0
        self.failed_interactions = 0
        
        # Load previous sessions and initialize the current one
        self.sessions = {}
        self.load_sessions()
        self.initialize_current_session()
        
        logger.info(f"Initialized session memory with ID {self.current_session_id}")
    
    def load_sessions(self):
        """Load session data from disk if available."""
        if not self.memory_path:
            return
            
        sessions_file = os.path.join(self.memory_path, "sessions.json")
        
        if os.path.exists(sessions_file):
            try:
                with open(sessions_file, 'r') as f:
                    self.sessions = json.load(f)
                    
                # Clean up expired sessions
                current_time = datetime.now()
                expiry_threshold = timedelta(hours=self.session_expiry_hours)
                
                expired_sessions = []
                for session_id, session_data in self.sessions.items():
                    session_time = datetime.fromisoformat(session_data.get('timestamp', '2000-01-01T00:00:00'))
                    if current_time - session_time > expiry_threshold:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    
                logger.info(f"Loaded {len(self.sessions)} active sessions, removed {len(expired_sessions)} expired sessions")
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
    
    def save_sessions(self):
        """Save session data to disk."""
        if not self.memory_path:
            return
            
        sessions_file = os.path.join(self.memory_path, "sessions.json")
        
        try:
            # Update the current session data
            self.sessions[self.current_session_id] = self.get_current_session_data()
            
            with open(sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2)
                
            logger.info(f"Saved {len(self.sessions)} sessions to {sessions_file}")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def initialize_current_session(self):
        """Initialize the current session data."""
        self.sessions[self.current_session_id] = {
            'session_id': self.current_session_id,
            'timestamp': self.session_start_time.isoformat(),
            'total_interactions': 0,
            'element_types': {},
            'detection_accuracy': {},
            'success_rate': 0.0
        }
        
        logger.info(f"Initialized session {self.current_session_id}")
    
    def get_current_session_data(self):
        """Get the current session data."""
        success_rate = 0.0
        if self.total_interactions > 0:
            success_rate = self.successful_interactions / self.total_interactions
            
        return {
            'session_id': self.current_session_id,
            'timestamp': self.session_start_time.isoformat(),
            'total_interactions': self.total_interactions,
            'successful_interactions': self.successful_interactions,
            'failed_interactions': self.failed_interactions,
            'element_interactions': dict(self.element_interactions),
            'element_successes': dict(self.element_successes),
            'element_failures': dict(self.element_failures),
            'detection_counts': dict(self.detection_counts),
            'success_rate': success_rate,
            'interaction_sequence_length': len(self.interaction_sequence),
            'detection_accuracy': {
                element_type: np.mean(accuracies) if accuracies else 0.0
                for element_type, accuracies in self.detection_accuracy.items()
            }
        }
    
    def record_interaction(self, element_type: str, success: bool, screen_location: Tuple[int, int] = None):
        """Record a user interaction with a UI element."""
        # Update interaction counts
        self.element_interactions[element_type] += 1
        self.total_interactions += 1
        
        if success:
            self.element_successes[element_type] += 1
            self.successful_interactions += 1
        else:
            self.element_failures[element_type] += 1
            self.failed_interactions += 1
        
        # Record interaction in sequence
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'element_type': element_type,
            'success': success
        }
        
        if screen_location:
            x, y = screen_location
            interaction['location'] = {'x': x, 'y': y}
            
            # Update location heatmap (divide screen into 10x10 grid)
            grid_x = x // 100  # Assuming screen width of 1000 pixels
            grid_y = y // 100  # Assuming screen height of 1000 pixels
            grid_location = f"{grid_x},{grid_y}"
            self.location_heatmap[grid_location] += 1
        
        self.interaction_sequence.append(interaction)
        
        logger.info(f"Recorded {element_type} interaction, success={success}")
        
        # Save sessions after every 5 interactions
        if self.total_interactions % 5 == 0:
            self.save_sessions()
    
    def record_detection(self, element_type: str, accuracy: float = None):
        """Record a detection of a UI element with optional accuracy."""
        self.detection_counts[element_type] += 1
        
        if accuracy is not None:
            self.detection_accuracy[element_type].append(accuracy)
        
        logger.info(f"Recorded detection of {element_type}, accuracy={accuracy}")
    
    def get_prioritized_element_types(self, limit: int = 5) -> List[str]:
        """Get a list of element types prioritized by interaction frequency."""
        # Sort element types by interaction count
        sorted_elements = sorted(
            self.element_interactions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N element types
        return [element_type for element_type, _ in sorted_elements[:limit]]
    
    def get_success_rates(self) -> Dict[str, float]:
        """Get success rates for each element type."""
        success_rates = {}
        
        for element_type, count in self.element_interactions.items():
            if count > 0:
                success_rate = self.element_successes[element_type] / count
                success_rates[element_type] = success_rate
        
        return success_rates
    
    def get_interaction_hotspots(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get the top N interaction hotspots on the screen."""
        sorted_locations = sorted(
            self.location_heatmap.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        hotspots = []
        for location_str, count in sorted_locations[:top_n]:
            x_grid, y_grid = map(int, location_str.split(','))
            hotspots.append({
                'grid_location': {'x': x_grid, 'y': y_grid},
                'center_point': {'x': x_grid * 100 + 50, 'y': y_grid * 100 + 50},
                'interaction_count': count
            })
        
        return hotspots
    
    def get_interaction_patterns(self) -> List[Dict[str, Any]]:
        """Analyze interaction sequence to find patterns."""
        if len(self.interaction_sequence) < 3:
            return []
        
        patterns = []
        sequence_length = len(self.interaction_sequence)
        
        # Find common bigrams and trigrams in the interaction sequence
        bigrams = defaultdict(int)
        trigrams = defaultdict(int)
        
        for i in range(sequence_length - 1):
            current = self.interaction_sequence[i]['element_type']
            next_element = self.interaction_sequence[i + 1]['element_type']
            bigram = f"{current}->{next_element}"
            bigrams[bigram] += 1
            
            if i < sequence_length - 2:
                next_next = self.interaction_sequence[i + 2]['element_type']
                trigram = f"{current}->{next_element}->{next_next}"
                trigrams[trigram] += 1
        
        # Get top patterns
        top_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:5]
        top_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for pattern, count in top_bigrams:
            if count >= 2:  # Only report patterns that occur at least twice
                patterns.append({
                    'pattern': pattern,
                    'count': count,
                    'type': 'bigram'
                })
                
        for pattern, count in top_trigrams:
            if count >= 2:
                patterns.append({
                    'pattern': pattern,
                    'count': count,
                    'type': 'trigram'
                })
        
        return patterns
    
    def enhance_detection_priority(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance detection priority based on session history."""
        if not detections:
            return detections
            
        # Get prioritized element types
        priority_types = self.get_prioritized_element_types()
        
        # Get interaction hotspots
        hotspots = self.get_interaction_hotspots(5)
        hotspot_regions = [
            (h['center_point']['x'] - 100, h['center_point']['y'] - 100,
             h['center_point']['x'] + 100, h['center_point']['y'] + 100)
            for h in hotspots
        ]
        
        # Calculate success rates
        success_rates = self.get_success_rates()
        
        enhanced_detections = []
        for detection in detections:
            element_type = detection.get('type')
            
            # Start with original confidence
            confidence = detection.get('confidence', 0.7)
            priority_boost = 0.0
            
            # Boost based on prioritized element types
            if element_type in priority_types:
                priority_index = priority_types.index(element_type)
                type_boost = 0.15 * (1 - priority_index / len(priority_types))
                priority_boost += type_boost
            
            # Boost based on hotspot locations
            if 'x' in detection and 'y' in detection:
                x, y = detection['x'], detection['y']
                for x1, y1, x2, y2 in hotspot_regions:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        priority_boost += 0.1
                        break
            
            # Adjust based on success rate
            if element_type in success_rates:
                success_rate = success_rates[element_type]
                success_boost = 0.05 * success_rate
                priority_boost += success_boost
            
            # Apply the total boost, capped at 25%
            priority_boost = min(priority_boost, 0.25)
            enhanced_confidence = min(confidence + priority_boost, 0.99)
            
            # Create enhanced detection
            enhanced_detection = detection.copy()
            enhanced_detection['confidence'] = enhanced_confidence
            enhanced_detection['priority_boost'] = priority_boost
            enhanced_detection['session_priority'] = priority_types.index(element_type) if element_type in priority_types else 999
            
            enhanced_detections.append(enhanced_detection)
        
        # Sort by priority and confidence
        enhanced_detections.sort(key=lambda d: (d.get('session_priority', 999), -d.get('confidence', 0)))
        
        logger.info(f"Enhanced detection priority for {len(enhanced_detections)} elements")
        return enhanced_detections
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        # Calculate success rate
        overall_success_rate = 0.0
        if self.total_interactions > 0:
            overall_success_rate = self.successful_interactions / self.total_interactions
        
        # Calculate element-specific metrics
        element_stats = {}
        for element_type in self.element_interactions.keys():
            interactions = self.element_interactions[element_type]
            successes = self.element_successes[element_type]
            success_rate = successes / interactions if interactions > 0 else 0.0
            
            detection_accuracy = 0.0
            if element_type in self.detection_accuracy and self.detection_accuracy[element_type]:
                detection_accuracy = np.mean(self.detection_accuracy[element_type])
            
            element_stats[element_type] = {
                'interactions': interactions,
                'success_rate': success_rate,
                'detection_count': self.detection_counts[element_type],
                'detection_accuracy': detection_accuracy
            }
        
        # Get interaction patterns
        patterns = self.get_interaction_patterns()
        
        return {
            'session_id': self.current_session_id,
            'duration': (datetime.now() - self.session_start_time).total_seconds() / 60,  # in minutes
            'total_interactions': self.total_interactions,
            'overall_success_rate': overall_success_rate,
            'element_stats': element_stats,
            'prioritized_elements': self.get_prioritized_element_types(),
            'interaction_patterns': patterns,
            'interaction_hotspots': self.get_interaction_hotspots(),
            'timestamp': datetime.now().isoformat()
        }
