#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Self-expanding template library for UI detection.
This module allows the system to automatically learn from successful detections
and add them to the template library for future reference.
"""

import os
import json
import uuid
import numpy as np
import cv2
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple, Optional

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

# Template library settings
TEMPLATE_DIR = Path('ui_templates')
LEARNED_DIR = TEMPLATE_DIR / 'learned'
TEMPLATE_METADATA_FILE = TEMPLATE_DIR / 'template_metadata.json'

class SelfExpandingTemplateLibrary:
    """
    A self-expanding library of UI element templates that learns from 
    successful detections and adds them to improve future detection accuracy.
    """
    
    def __init__(self, memory_path: str = None) -> None:
        """Initialize the self-expanding template library.
        
        Args:
            memory_path: Optional custom path for storing template metadata
        """
        # Ensure directories exist
        self.template_dir = TEMPLATE_DIR
        self.learned_dir = LEARNED_DIR
        os.makedirs(self.learned_dir, exist_ok=True)
        
        # Create subdirectories for element types
        for element_type in ['button', 'text_input', 'checkbox', 'icon', 'dropdown', 'toggle']:
            os.makedirs(self.learned_dir / element_type, exist_ok=True)
        
        # Load or create template metadata
        self.metadata_file = Path(memory_path) / 'template_metadata.json' if memory_path else TEMPLATE_METADATA_FILE
        self.metadata = self._load_metadata()
        
        # Track detection statistics for each template
        self.detection_stats = self.metadata.get('detection_stats', {})
        
        # Track quality score for each template
        self.quality_scores = self.metadata.get('quality_scores', {})
        
        # Initialize expansion counters
        self.templates_added = 0
        self.templates_improved = 0
        
        logger.info(f"Initialized self-expanding template library at {self.template_dir}")

    def _load_metadata(self) -> Dict:
        """Load template metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading template metadata: {e}")
                return self._create_default_metadata()
        else:
            logger.info(f"Creating new template metadata file at {self.metadata_file}")
            return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict:
        """Create default template metadata structure."""
        return {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'templates': {},
            'detection_stats': {},
            'quality_scores': {},
            'element_types': [
                'button', 'text_input', 'checkbox', 'radio', 'toggle',
                'icon', 'dropdown', 'image', 'link', 'menu'
            ],
            'contexts': []
        }
    
    def _save_metadata(self) -> None:
        """Save template metadata to JSON file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['detection_stats'] = self.detection_stats
        self.metadata['quality_scores'] = self.quality_scores
        
        try:
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved template metadata with {len(self.metadata['templates'])} templates")
        except IOError as e:
            logger.error(f"Error saving template metadata: {e}")
    
    def add_template(self, image: np.ndarray, element_type: str, text: str = '', 
                  context_info: Dict = None, confidence: float = 0.9) -> str:
        """Add a template directly to the library.
        
        Args:
            image: The UI element image
            element_type: Type of UI element (button, text_input, etc.)
            text: Text content associated with the element
            context_info: Additional context information
            confidence: Confidence in the template quality
            
        Returns:
            template_id: ID of the created or updated template
        """
        # Check if this is a valid element type
        if element_type not in self.metadata['element_types']:
            logger.warning(f"Unknown element type: {element_type}, adding to registry")
            self.metadata['element_types'].append(element_type)
            
        # Check for similar existing templates
        template_id = self._find_similar_template(image, element_type)
        
        contexts = [context_info.get('form', '')] if context_info and 'form' in context_info else []
        
        if template_id:
            # Update existing template
            self.metadata['templates'][template_id]['occurrences'] += 1
            
            # Update contexts if needed
            for context in contexts:
                if context and context not in self.metadata['templates'][template_id].get('contexts', []):
                    if 'contexts' not in self.metadata['templates'][template_id]:
                        self.metadata['templates'][template_id]['contexts'] = []
                    self.metadata['templates'][template_id]['contexts'].append(context)
                    
            # Update metadata
            self.metadata['templates'][template_id]['last_updated'] = datetime.now().isoformat()
            self.templates_improved += 1
            
            logger.info(f"Updated existing template {template_id} for {element_type}")
        else:
            # Create new template ID
            template_id = str(uuid.uuid4())
            
            # Prepare template metadata
            template_metadata = {
                'id': template_id,
                'element_type': element_type,
                'text': text,
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'occurrences': 1,
                'source': 'manual'
            }
            
            # Add contexts if available
            if contexts:
                template_metadata['contexts'] = contexts
                
            # Add additional context info
            if context_info:
                for key, value in context_info.items():
                    if key != 'form':
                        template_metadata[key] = value
            
            # Add to metadata
            self.metadata['templates'][template_id] = template_metadata
            
            # Initialize detection stats and quality score
            self.detection_stats[template_id] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0
            }
            
            self.quality_scores[template_id] = confidence
            self.templates_added += 1
            
            # Save the image file
            element_dir = self.learned_dir / element_type
            os.makedirs(element_dir, exist_ok=True)
            image_path = element_dir / f"{template_id}.png"
            
            cv2.imwrite(str(image_path), image)
            logger.info(f"Added new template {template_id} for {element_type}")
        
        # Save metadata
        self._save_metadata()
        
        return template_id

    def add_successful_detection(self, image: np.ndarray, element_data: Dict, 
                               source_detector: str, confidence: float, 
                               context: str = None) -> str:
        """Add a successful detection as a new template.
        
        Args:
            image: The detected UI element image
            element_data: Dictionary containing element properties
            source_detector: The detector that successfully found this element
            confidence: Detection confidence score
            context: Optional context where this element was detected
            
        Returns:
            template_id: Unique ID of the added/updated template
        """
        element_type = element_data.get('type', 'unknown')
        rect = element_data.get('rect', {})
        
        # Extract the element from the image
        x = max(0, rect.get('x', 0))
        y = max(0, rect.get('y', 0))
        w = min(rect.get('width', 0), image.shape[1] - x)
        h = min(rect.get('height', 0), image.shape[0] - y)
        
        # Skip if dimensions are invalid
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid dimensions for element extraction: {w}x{h}")
            return None
            
        # Extract the element image
        element_img = image[y:y+h, x:x+w].copy()
        
        # Check if this element is too similar to existing templates
        similar_template = self._find_similar_template(element_img, element_type)
        
        if similar_template:
            # Update the existing template statistics
            template_id = similar_template
            self._update_template_stats(template_id, confidence, True)
            self.templates_improved += 1
            logger.info(f"Updated similar template {template_id} for {element_type}")
            return template_id
        
        # Generate a unique ID for this template
        template_id = f"{element_type}_{str(uuid.uuid4())[:8]}"
        
        # Create template metadata
        template_metadata = {
            'id': template_id,
            'element_type': element_type,
            'source_detector': source_detector,
            'initial_confidence': confidence,
            'created': datetime.now().isoformat(),
            'dimensions': {'width': w, 'height': h},
            'properties': {k: v for k, v in element_data.items() 
                          if k not in ['rect', 'type']},
            'usage_count': 0,
            'success_count': 0
        }
        
        # Add context information if provided
        if context:
            template_metadata['contexts'] = [context]
            # Add to global contexts list if not already present
            if context not in self.metadata['contexts']:
                self.metadata['contexts'].append(context)
        
        # Save the template image
        template_path = self.learned_dir / element_type / f"{template_id}.png"
        cv2.imwrite(str(template_path), element_img)
        
        # Add to metadata
        self.metadata['templates'][template_id] = template_metadata
        self.detection_stats[template_id] = {
            'attempts': 0,
            'successes': 0,
            'last_used': None,
            'contexts': {}
        }
        self.quality_scores[template_id] = confidence
        
        # Save metadata changes
        self._save_metadata()
        
        self.templates_added += 1
        logger.info(f"Added new template {template_id} for {element_type} with confidence {confidence:.2f}")
        
        return template_id
    
    def _find_similar_template(self, element_img: np.ndarray, element_type: str) -> Optional[str]:
        """Find if there's a very similar template already in the library.
        
        Args:
            element_img: The element image to check
            element_type: The type of UI element
            
        Returns:
            template_id: ID of similar template if found, None otherwise
        """
        # Only search templates of the same element type
        template_dir = self.learned_dir / element_type
        if not template_dir.exists():
            return None
            
        best_match = None
        best_score = 0.8  # Threshold for considering templates similar
        
        # Use template matching to find similar templates
        for template_file in template_dir.glob("*.png"):
            template_id = template_file.stem
            template = cv2.imread(str(template_file))
            
            if template is None or template.shape[0] > element_img.shape[0] or template.shape[1] > element_img.shape[1]:
                continue
                
            try:
                # Resize template to match the element's dimensions for better comparison
                template_resized = cv2.resize(template, (element_img.shape[1], element_img.shape[0]))
                
                # Calculate similarity
                result = cv2.matchTemplate(element_img, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = template_id
            except Exception as e:
                logger.warning(f"Error comparing templates: {e}")
                continue
        
        return best_match
    
    def _update_template_stats(self, template_id: str, confidence: float, success: bool) -> None:
        """Update usage statistics for a template.
        
        Args:
            template_id: The template ID
            confidence: Detection confidence
            success: Whether detection was successful
        """
        if template_id not in self.detection_stats:
            self.detection_stats[template_id] = {
                'attempts': 0,
                'successes': 0,
                'last_used': None,
                'contexts': {}
            }
            
        self.detection_stats[template_id]['attempts'] += 1
        self.detection_stats[template_id]['last_used'] = datetime.now().isoformat()
        
        if success:
            self.detection_stats[template_id]['successes'] += 1
            
            # Update quality score with exponential moving average
            current_score = self.quality_scores.get(template_id, 0.5)
            alpha = 0.3  # Weight for new observation
            new_score = (1 - alpha) * current_score + alpha * confidence
            self.quality_scores[template_id] = new_score
    
    def get_templates_for_element_type(self, element_type: str, context: str = None) -> List[Dict]:
        """Get all templates for a specific element type.
        
        Args:
            element_type: Type of UI element
            context: Optional context to filter templates
            
        Returns:
            List of template metadata dictionaries with additional path info
        """
        templates = []
        
        for template_id, metadata in self.metadata['templates'].items():
            if metadata['element_type'] == element_type:
                # Filter by context if provided
                if context and 'contexts' in metadata and context not in metadata['contexts']:
                    continue
                    
                # Add template path to metadata
                template_info = metadata.copy()
                template_info['path'] = str(self.learned_dir / element_type / f"{template_id}.png")
                
                # Add quality score
                template_info['quality_score'] = self.quality_scores.get(template_id, 0.5)
                
                templates.append(template_info)
        
        # Sort by quality score in descending order
        templates.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return templates
    
    def provide_feedback(self, template_id: str, context: str, success: bool) -> None:
        """Provide feedback about template detection success/failure.
        
        Args:
            template_id: The template ID
            context: The context in which detection occurred
            success: Whether detection was successful
        """
        if template_id not in self.detection_stats:
            return
            
        # Update context-specific statistics
        if 'contexts' not in self.detection_stats[template_id]:
            self.detection_stats[template_id]['contexts'] = {}
            
        if context not in self.detection_stats[template_id]['contexts']:
            self.detection_stats[template_id]['contexts'][context] = {
                'attempts': 0,
                'successes': 0
            }
            
        self.detection_stats[template_id]['contexts'][context]['attempts'] += 1
        
        if success:
            self.detection_stats[template_id]['contexts'][context]['successes'] += 1
            
        # Update template metadata contexts
        if success and template_id in self.metadata['templates']:
            if 'contexts' not in self.metadata['templates'][template_id]:
                self.metadata['templates'][template_id]['contexts'] = []
                
            if context not in self.metadata['templates'][template_id]['contexts']:
                self.metadata['templates'][template_id]['contexts'].append(context)
        
        # Save metadata
        self._save_metadata()
        
    def get_expansion_stats(self) -> Dict:
        """Get statistics about template library expansion.
        
        Returns:
            Dictionary with expansion statistics
        """
        return {
            'templates_added': self.templates_added,
            'templates_improved': self.templates_improved,
            'total_templates': len(self.metadata['templates']),
            'element_types': {
                element_type: len([t for t in self.metadata['templates'].values() 
                                 if t['element_type'] == element_type])
                for element_type in self.metadata['element_types']
                if any(t['element_type'] == element_type for t in self.metadata['templates'].values())
            }
        }
        
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the template library.
        
        Returns:
            Dictionary with detailed template library statistics and performance metrics
        """
        expansion_stats = self.get_expansion_stats()
        
        # Calculate template usage statistics
        total_attempts = sum(stats.get('attempts', 0) for stats in self.detection_stats.values())
        total_successes = sum(stats.get('successes', 0) for stats in self.detection_stats.values())
        total_failures = sum(stats.get('failures', 0) for stats in self.detection_stats.values())
        
        # Calculate per-context statistics
        context_stats = {}
        for template_id, stats in self.detection_stats.items():
            if 'contexts' not in stats:
                continue
                
            for context, context_data in stats['contexts'].items():
                if context not in context_stats:
                    context_stats[context] = {
                        'attempts': 0,
                        'successes': 0,
                        'templates': 0
                    }
                    
                context_stats[context]['attempts'] += context_data.get('attempts', 0)
                context_stats[context]['successes'] += context_data.get('successes', 0)
                context_stats[context]['templates'] += 1
        
        # Calculate average quality score per element type
        quality_by_type = {}
        for template_id, quality in self.quality_scores.items():
            if template_id not in self.metadata['templates']:
                continue
                
            element_type = self.metadata['templates'][template_id]['element_type']
            if element_type not in quality_by_type:
                quality_by_type[element_type] = []
                
            quality_by_type[element_type].append(quality)
        
        avg_quality_by_type = {
            element_type: sum(scores) / len(scores) if scores else 0.0
            for element_type, scores in quality_by_type.items()
        }
        
        # Get the most recently added templates
        recent_templates = []
        if self.metadata['templates']:
            templates = [(tid, tdata) for tid, tdata in self.metadata['templates'].items()]
            # Sort by creation date, latest first
            templates.sort(key=lambda x: x[1].get('created', ''), reverse=True)
            recent_templates = [t[0] for t in templates[:5]]  # Get 5 most recent
        
        # Build comprehensive statistics
        return {
            **expansion_stats,
            'usage_stats': {
                'total_attempts': total_attempts,
                'total_successes': total_successes,
                'total_failures': total_failures,
                'success_rate': total_successes / total_attempts if total_attempts > 0 else 0.0
            },
            'context_stats': context_stats,
            'quality_metrics': {
                'avg_quality_by_type': avg_quality_by_type,
                'overall_avg_quality': sum(self.quality_scores.values()) / len(self.quality_scores) if self.quality_scores else 0.0
            },
            'recent_templates': recent_templates,
            'learning_progress': {
                'templates_per_type': expansion_stats['element_types'],
                'adaptive_growth': self.templates_added > 0,
                'template_diversity': len(expansion_stats['element_types'])
            }
        }

def create_template_library(memory_path: str = None) -> SelfExpandingTemplateLibrary:
    """Create and initialize the self-expanding template library.
    
    Args:
        memory_path: Optional path to store template metadata
        
    Returns:
        Initialized SelfExpandingTemplateLibrary instance
    """
    return SelfExpandingTemplateLibrary(memory_path)
