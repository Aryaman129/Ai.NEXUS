#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Context Learner for UI detection.
This module implements transfer learning across different application contexts
to improve detection by sharing knowledge between different domains.
"""

import os
import json
import uuid
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

# Configure logging
logger = logging.getLogger("adaptive_ui_detection")

class CrossContextLearner:
    """
    Learns patterns across different UI contexts and transfers knowledge between them.
    This enables the detection system to adapt to new applications based on knowledge
    from previously seen applications.
    """
    
    def __init__(self, memory_path: str = None):
        """Initialize the cross-context learner.
        
        Args:
            memory_path: Path to store context knowledge data
        """
        self.name = "CrossContextLearner"
        
        # Set up paths
        self.contexts_dir = Path(memory_path) if memory_path else Path('context_knowledge')
        os.makedirs(self.contexts_dir, exist_ok=True)
        
        # Knowledge file
        self.knowledge_file = self.contexts_dir / 'context_knowledge.json'
        
        # Load or initialize knowledge base
        self.knowledge = self._load_knowledge()
        
        # Transfer count
        self.transfers_performed = 0
        self.knowledge_items_shared = 0
        
        logger.info(f"Initialized cross-context learner with {len(self.knowledge['contexts'])} contexts")
    
    def _load_knowledge(self) -> Dict:
        """Load knowledge base from file.
        
        Returns:
            Knowledge base dictionary
        """
        default_knowledge = {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'contexts': {},
            'element_type_patterns': {},
            'cross_context_matches': {},
            'context_similarity': {}
        }
        
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'r') as f:
                    knowledge = json.load(f)
                logger.info(f"Loaded cross-context knowledge from {self.knowledge_file}")
                return knowledge
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading knowledge base: {e}")
                return default_knowledge
        
        logger.info(f"Creating new cross-context knowledge base at {self.knowledge_file}")
        return default_knowledge
    
    def _save_knowledge(self) -> None:
        """Save knowledge base to file."""
        self.knowledge['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
            logger.info(f"Saved cross-context knowledge with {len(self.knowledge['contexts'])} contexts")
        except IOError as e:
            logger.error(f"Error saving knowledge base: {e}")
    
    def add_context_knowledge(self, context_name: str, element_data: Dict, screenshot: np.ndarray = None) -> None:
        """Add element knowledge for a specific context.
        
        Args:
            context_name: Name of the application/context (e.g., 'gmail', 'excel')
            element_data: Data for the UI element
            screenshot: Optional screenshot where the element was found
        """
        if not context_name or not element_data:
            return
            
        # Initialize context if it doesn't exist
        if context_name not in self.knowledge['contexts']:
            self.knowledge['contexts'][context_name] = {
                'elements': {},
                'patterns': {},
                'creation_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        # Extract element type
        element_type = element_data.get('type', 'unknown')
        if element_type == 'unknown':
            return
            
        # Generate a unique ID for the element
        element_id = str(uuid.uuid4())
        
        # Extract visual characteristics if screenshot available
        visual_characteristics = {}
        if screenshot is not None and 'rect' in element_data:
            try:
                rect = element_data['rect']
                x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
                
                if x < 0 or y < 0 or x + w > screenshot.shape[1] or y + h > screenshot.shape[0]:
                    # Invalid coordinates
                    pass
                else:
                    # Extract element image
                    element_img = screenshot[y:y+h, x:x+w]
                    
                    # Calculate color histogram
                    if len(element_img.shape) == 3:  # Color image
                        hist = cv2.calcHist([element_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten().tolist()
                    else:  # Grayscale
                        hist = cv2.calcHist([element_img], [0], None, [16], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten().tolist()
                        
                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Analyze text characteristics
                    text_length = len(element_data.get('text', ''))
                    has_text = text_length > 0
                    
                    # Store visual characteristics
                    visual_characteristics = {
                        'color_histogram': hist,
                        'aspect_ratio': aspect_ratio,
                        'size': w * h,
                        'has_text': has_text,
                        'text_length': text_length
                    }
                    
                    # Save element image for future reference
                    element_dir = self.contexts_dir / context_name / element_type
                    os.makedirs(element_dir, exist_ok=True)
                    cv2.imwrite(str(element_dir / f"{element_id}.png"), element_img)
                    
                    # Store path to the saved image
                    visual_characteristics['image_path'] = str(element_dir / f"{element_id}.png")
            except Exception as e:
                logger.warning(f"Error extracting visual characteristics: {e}")
        
        # Create element knowledge entry
        element_data_copy = element_data.copy()
        if 'id' in element_data_copy:
            element_data_copy['original_id'] = element_data_copy.pop('id')
            
        element_knowledge = {
            'id': element_id,
            'element_data': element_data_copy,
            'visual_characteristics': visual_characteristics,
            'creation_date': datetime.now().isoformat()
        }
        
        # Add to context knowledge
        self.knowledge['contexts'][context_name]['elements'][element_id] = element_knowledge
        
        # Update context's last updated timestamp
        self.knowledge['contexts'][context_name]['last_updated'] = datetime.now().isoformat()
        
        # Update element type patterns
        self._update_element_type_patterns(context_name, element_type, element_knowledge)
        
        # Save knowledge base
        self._save_knowledge()
        
        logger.info(f"Added {element_type} element knowledge to {context_name} context")
    
    def _update_element_type_patterns(self, context_name: str, element_type: str, element_knowledge: Dict) -> None:
        """Update element type patterns for a context.
        
        Args:
            context_name: Context name
            element_type: Element type
            element_knowledge: Element knowledge data
        """
        # Initialize element type in patterns if needed
        if element_type not in self.knowledge['element_type_patterns']:
            self.knowledge['element_type_patterns'][element_type] = {
                'contexts': set(),
                'visual_patterns': {},
                'context_specific_patterns': {}
            }
            
        # Add context to the element type's contexts
        if isinstance(self.knowledge['element_type_patterns'][element_type]['contexts'], set):
            self.knowledge['element_type_patterns'][element_type]['contexts'].add(context_name)
            # Convert set to list for JSON serialization
            self.knowledge['element_type_patterns'][element_type]['contexts'] = list(
                self.knowledge['element_type_patterns'][element_type]['contexts']
            )
        elif isinstance(self.knowledge['element_type_patterns'][element_type]['contexts'], list):
            if context_name not in self.knowledge['element_type_patterns'][element_type]['contexts']:
                self.knowledge['element_type_patterns'][element_type]['contexts'].append(context_name)
        
        # Initialize context-specific patterns for this element type
        if context_name not in self.knowledge['element_type_patterns'][element_type]['context_specific_patterns']:
            self.knowledge['element_type_patterns'][element_type]['context_specific_patterns'][context_name] = {
                'aspect_ratio_range': [999, 0],  # [min, max]
                'size_range': [999999, 0],       # [min, max]
                'text_length_range': [999, 0],   # [min, max]
                'color_histograms': [],
                'element_count': 0
            }
        
        context_pattern = self.knowledge['element_type_patterns'][element_type]['context_specific_patterns'][context_name]
        
        # Update pattern with this element's characteristics
        visual_chars = element_knowledge.get('visual_characteristics', {})
        if visual_chars:
            # Update aspect ratio range
            aspect_ratio = visual_chars.get('aspect_ratio', 0)
            if aspect_ratio > 0:
                context_pattern['aspect_ratio_range'][0] = min(context_pattern['aspect_ratio_range'][0], aspect_ratio)
                context_pattern['aspect_ratio_range'][1] = max(context_pattern['aspect_ratio_range'][1], aspect_ratio)
                
            # Update size range
            size = visual_chars.get('size', 0)
            if size > 0:
                context_pattern['size_range'][0] = min(context_pattern['size_range'][0], size)
                context_pattern['size_range'][1] = max(context_pattern['size_range'][1], size)
                
            # Update text length range
            text_length = visual_chars.get('text_length', 0)
            context_pattern['text_length_range'][0] = min(context_pattern['text_length_range'][0], text_length)
            context_pattern['text_length_range'][1] = max(context_pattern['text_length_range'][1], text_length)
            
            # Store color histogram (limit to 10 for memory efficiency)
            color_hist = visual_chars.get('color_histogram', [])
            if color_hist and len(context_pattern['color_histograms']) < 10:
                context_pattern['color_histograms'].append(color_hist)
                
        # Increment element count
        context_pattern['element_count'] += 1
        
        # Initialize context in patterns if needed
        if context_name not in self.knowledge['contexts'][context_name]['patterns']:
            self.knowledge['contexts'][context_name]['patterns'][element_type] = {
                'count': 0,
                'examples': []
            }
            
        # Update context's patterns for this element type
        context_elem_pattern = self.knowledge['contexts'][context_name]['patterns'][element_type]
        context_elem_pattern['count'] += 1
        
        # Add example if we don't have too many yet
        if len(context_elem_pattern['examples']) < 5:
            context_elem_pattern['examples'].append(element_id)
    
    def find_similar_elements(self, query_element: Dict, context_name: str = None) -> List[Dict]:
        """Find elements similar to the query element, potentially across contexts.
        
        Args:
            query_element: Element data to find similar elements for
            context_name: Optional context to constrain the search to
            
        Returns:
            List of similar elements with similarity scores
        """
        if not query_element:
            return []
            
        element_type = query_element.get('type', 'unknown')
        if element_type == 'unknown':
            return []
            
        # Collect all candidate elements of this type
        candidate_elements = []
        
        # If context is specified, prioritize elements from that context
        if context_name and context_name in self.knowledge['contexts']:
            for element_id, element in self.knowledge['contexts'][context_name]['elements'].items():
                if element['element_data'].get('type') == element_type:
                    candidate_elements.append((context_name, element))
        
        # Also include elements from other contexts
        for ctx_name, context in self.knowledge['contexts'].items():
            if ctx_name == context_name:
                continue  # Already processed above
                
            for element_id, element in context['elements'].items():
                if element['element_data'].get('type') == element_type:
                    candidate_elements.append((ctx_name, element))
        
        # Calculate similarity for each candidate
        similar_elements = []
        for ctx_name, element in candidate_elements:
            similarity_score = self._calculate_element_similarity(query_element, element['element_data'])
            if similarity_score > 0.6:  # Only include if reasonably similar
                similar_elements.append({
                    'context': ctx_name,
                    'element': element['element_data'],
                    'similarity': similarity_score,
                    'id': element['id']
                })
                
        # Sort by similarity (highest first)
        similar_elements.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit to top 10 for efficiency
        return similar_elements[:10]
    
    def _calculate_element_similarity(self, element1: Dict, element2: Dict) -> float:
        """Calculate similarity between two elements (0 to 1).
        
        Args:
            element1: First element data
            element2: Second element data
            
        Returns:
            Similarity score between 0 and 1
        """
        # Start with a base similarity score
        similarity = 0.5
        
        # Same type is a strong indicator
        if element1.get('type') == element2.get('type'):
            similarity += 0.3
        else:
            return 0.0  # Different types, not similar
            
        # Similar text content is a strong indicator
        text1 = element1.get('text', '').lower()
        text2 = element2.get('text', '').lower()
        
        if text1 and text2:
            # Check for exact match
            if text1 == text2:
                similarity += 0.2
            # Check for substring
            elif text1 in text2 or text2 in text1:
                similarity += 0.1
            # Check for word overlap
            else:
                words1 = set(text1.split())
                words2 = set(text2.split())
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
                    similarity += 0.1 * overlap
        
        # Similar role/function
        if element1.get('role') == element2.get('role'):
            similarity += 0.1
            
        # Normalize to ensure it's between 0 and 1
        return min(1.0, max(0.0, similarity))
    
    def transfer_knowledge(self, source_context: str, target_context: str, element_type: str = None) -> Dict:
        """Transfer knowledge from source context to target context.
        
        Args:
            source_context: Context to transfer knowledge from
            target_context: Context to transfer knowledge to
            element_type: Optional element type to limit transfer to
            
        Returns:
            Dictionary with transfer results
        """
        if source_context not in self.knowledge['contexts']:
            return {'success': False, 'message': f"Source context {source_context} not found"}
            
        # Create target context if it doesn't exist
        if target_context not in self.knowledge['contexts']:
            self.knowledge['contexts'][target_context] = {
                'elements': {},
                'patterns': {},
                'creation_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        # Track transfer statistics
        transfer_stats = {
            'source': source_context,
            'target': target_context,
            'element_types_transferred': [],
            'patterns_transferred': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Transfer element type patterns
        for elem_type, patterns in self.knowledge['element_type_patterns'].items():
            # Skip if element type filter is specified and doesn't match
            if element_type and elem_type != element_type:
                continue
                
            # Check if source context has patterns for this element type
            if source_context in patterns['context_specific_patterns']:
                source_pattern = patterns['context_specific_patterns'][source_context]
                
                # Only transfer if source has enough examples
                if source_pattern['element_count'] >= 3:
                    # Create target pattern if it doesn't exist
                    if target_context not in patterns['context_specific_patterns']:
                        patterns['context_specific_patterns'][target_context] = {
                            'aspect_ratio_range': source_pattern['aspect_ratio_range'].copy(),
                            'size_range': source_pattern['size_range'].copy(),
                            'text_length_range': source_pattern['text_length_range'].copy(),
                            'color_histograms': source_pattern['color_histograms'][:3] if source_pattern['color_histograms'] else [],  # Copy a few histograms
                            'element_count': 0  # Start with 0 count
                        }
                    else:
                        # Merge patterns
                        target_pattern = patterns['context_specific_patterns'][target_context]
                        
                        # Expand ranges to include source ranges
                        target_pattern['aspect_ratio_range'][0] = min(target_pattern['aspect_ratio_range'][0], source_pattern['aspect_ratio_range'][0])
                        target_pattern['aspect_ratio_range'][1] = max(target_pattern['aspect_ratio_range'][1], source_pattern['aspect_ratio_range'][1])
                        
                        target_pattern['size_range'][0] = min(target_pattern['size_range'][0], source_pattern['size_range'][0])
                        target_pattern['size_range'][1] = max(target_pattern['size_range'][1], source_pattern['size_range'][1])
                        
                        target_pattern['text_length_range'][0] = min(target_pattern['text_length_range'][0], source_pattern['text_length_range'][0])
                        target_pattern['text_length_range'][1] = max(target_pattern['text_length_range'][1], source_pattern['text_length_range'][1])
                        
                        # Add a few color histograms if we don't have many
                        if len(target_pattern['color_histograms']) < 5 and source_pattern['color_histograms']:
                            target_pattern['color_histograms'].extend(source_pattern['color_histograms'][:2])
                            
                    # Update statistics
                    transfer_stats['element_types_transferred'].append(elem_type)
                    transfer_stats['patterns_transferred'] += 1
                    
                    # Add transfer to cross-context matches
                    if 'transfers' not in self.knowledge['cross_context_matches']:
                        self.knowledge['cross_context_matches']['transfers'] = []
                        
                    self.knowledge['cross_context_matches']['transfers'].append({
                        'source': source_context,
                        'target': target_context,
                        'element_type': elem_type,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    })
                    
                    # Update context similarity
                    self._update_context_similarity(source_context, target_context)
                    
        # Update transfer counts
        self.transfers_performed += 1
        self.knowledge_items_shared += transfer_stats['patterns_transferred']
        
        # Save knowledge base
        self._save_knowledge()
        
        logger.info(f"Transferred knowledge from {source_context} to {target_context}: " +
                   f"{len(transfer_stats['element_types_transferred'])} element types")
        
        return {
            'success': True,
            'stats': transfer_stats
        }
    
    def _update_context_similarity(self, context1: str, context2: str) -> None:
        """Update similarity score between two contexts.
        
        Args:
            context1: First context
            context2: Second context
        """
        # Initialize context similarity dict if needed
        if 'context_pairs' not in self.knowledge['context_similarity']:
            self.knowledge['context_similarity']['context_pairs'] = {}
            
        # Create a consistent pair key (alphabetical order)
        context_pair = tuple(sorted([context1, context2]))
        pair_key = f"{context_pair[0]}___{context_pair[1]}"
        
        # Initialize similarity data for this pair if needed
        if pair_key not in self.knowledge['context_similarity']['context_pairs']:
            self.knowledge['context_similarity']['context_pairs'][pair_key] = {
                'contexts': list(context_pair),
                'similarity_score': 0.5,  # Start with neutral similarity
                'shared_element_types': [],
                'last_updated': datetime.now().isoformat()
            }
            
        pair_data = self.knowledge['context_similarity']['context_pairs'][pair_key]
        
        # Calculate number of shared element types
        element_types1 = set()
        if context1 in self.knowledge['contexts']:
            element_types1 = set(self.knowledge['contexts'][context1].get('patterns', {}).keys())
            
        element_types2 = set()
        if context2 in self.knowledge['contexts']:
            element_types2 = set(self.knowledge['contexts'][context2].get('patterns', {}).keys())
            
        shared_types = element_types1.intersection(element_types2)
        
        # Update shared element types
        pair_data['shared_element_types'] = list(shared_types)
        
        # Calculate new similarity score based on shared elements
        total_types = len(element_types1.union(element_types2))
        if total_types > 0:
            type_similarity = len(shared_types) / total_types
            # Blend with existing similarity (giving more weight to new data)
            pair_data['similarity_score'] = 0.3 * pair_data['similarity_score'] + 0.7 * type_similarity
            
        # Update timestamp
        pair_data['last_updated'] = datetime.now().isoformat()
    
    def get_most_similar_contexts(self, context_name: str, limit: int = 3) -> List[Dict]:
        """Get the most similar contexts to a given context.
        
        Args:
            context_name: Context to find similar contexts for
            limit: Maximum number of similar contexts to return
            
        Returns:
            List of similar contexts with similarity scores
        """
        if context_name not in self.knowledge['contexts']:
            return []
            
        # Collect similarity scores for all context pairs including this context
        similar_contexts = []
        
        if 'context_pairs' in self.knowledge['context_similarity']:
            for pair_key, pair_data in self.knowledge['context_similarity']['context_pairs'].items():
                if context_name in pair_data['contexts']:
                    # Get the other context in the pair
                    other_context = pair_data['contexts'][0] if pair_data['contexts'][1] == context_name else pair_data['contexts'][1]
                    
                    similar_contexts.append({
                        'context': other_context,
                        'similarity': pair_data['similarity_score'],
                        'shared_element_types': pair_data['shared_element_types']
                    })
        
        # Sort by similarity (highest first)
        similar_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Limit to requested number
        return similar_contexts[:limit]
    
    def suggest_context_transfer(self, context_name: str) -> List[Dict]:
        """Suggest contexts to transfer knowledge from for a given context.
        
        Args:
            context_name: Context to get transfer suggestions for
            
        Returns:
            List of suggested source contexts with reasons
        """
        # New contexts with few elements benefit most from transfer
        is_new_context = False
        element_count = 0
        
        if context_name in self.knowledge['contexts']:
            context = self.knowledge['contexts'][context_name]
            element_count = len(context['elements'])
            
            # Consider new if created in the last day or has few elements
            creation_date = datetime.fromisoformat(context.get('creation_date', datetime.now().isoformat()))
            time_diff = (datetime.now() - creation_date).total_seconds()
            is_new_context = time_diff < 86400 or element_count < 10
        else:
            # Context doesn't exist yet, definitely new
            is_new_context = True
        
        # Only suggest transfers for new contexts
        if not is_new_context and element_count >= 20:
            return []
            
        # Get similar contexts
        similar_contexts = self.get_most_similar_contexts(context_name)
        
        # Filter to contexts with sufficient elements and knowledge
        suggestions = []
        for similar in similar_contexts:
            source_context = similar['context']
            
            if source_context in self.knowledge['contexts']:
                source_element_count = len(self.knowledge['contexts'][source_context]['elements'])
                
                if source_element_count >= 15:
                    # Good candidate for transfer
                    suggestions.append({
                        'source_context': source_context,
                        'similarity': similar['similarity'],
                        'reason': f"Has {source_element_count} elements and {len(similar['shared_element_types'])} shared element types",
                        'confidence': 0.7 * similar['similarity'] + 0.3 * min(1.0, source_element_count / 50)
                    })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions
    
    def get_context_knowledge_stats(self) -> Dict:
        """Get statistics about the cross-context knowledge base.
        
        Returns:
            Dictionary with knowledge statistics
        """
        context_stats = {}
        for context_name, context in self.knowledge['contexts'].items():
            context_stats[context_name] = {
                'element_count': len(context['elements']),
                'element_types': list(context.get('patterns', {}).keys()),
                'creation_date': context.get('creation_date', ''),
                'last_updated': context.get('last_updated', '')
            }
            
        return {
            'context_count': len(self.knowledge['contexts']),
            'element_type_count': len(self.knowledge['element_type_patterns']),
            'transfers_performed': self.transfers_performed,
            'knowledge_items_shared': self.knowledge_items_shared,
            'contexts': context_stats,
            'element_types': list(self.knowledge['element_type_patterns'].keys())
        }
