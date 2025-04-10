"""
Unit tests for visual memory system module

These tests verify that the visual memory system correctly stores, retrieves,
and learns from visual patterns related to UI elements.
"""
import os
import time
import json
import pytest
import numpy as np
import pickle
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem

# Create test data
def create_test_visual_signature():
    """Create a test visual signature (normalized 64x64 array)"""
    # Create a simple gradient pattern
    sig = np.zeros((64, 64))
    for i in range(64):
        for j in range(64):
            sig[i, j] = (i + j) / (64 + 64)
    return sig

def create_test_pattern(pattern_id=None):
    """Create a test UI element pattern"""
    pattern = {
        "visual_signature": create_test_visual_signature(),
        "type": "button",
        "text": "Test Button",
        "bbox_size": (100, 40),
        "metadata": {
            "context": {
                "window_title": "Test Window",
                "application": "TestApp"
            },
            "detection_confidence": 0.85
        }
    }
    
    if pattern_id:
        pattern["id"] = pattern_id
        
    return pattern


@pytest.fixture
def memory_system():
    """Create a VisualMemorySystem instance with temporary storage"""
    # Create a temporary directory for test memory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Create memory system
    memory = VisualMemorySystem(config={
        "memory_path": temp_dir,
        "max_patterns": 100,
        "similarity_threshold": 0.7,
        "enable_learning": True
    })
    
    yield memory
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)


def test_store_pattern(memory_system):
    """Test storing a pattern in visual memory"""
    # Create a test pattern
    pattern = create_test_pattern()
    
    # Store the pattern
    pattern_id = memory_system.store_pattern(pattern)
    
    # Check that a pattern ID was returned
    assert pattern_id is not None
    assert "pattern_" in pattern_id
    
    # Check that the pattern was stored
    assert len(memory_system.patterns) == 1
    
    # Check that the pattern contains the expected data
    stored_pattern = memory_system.patterns[0]
    assert stored_pattern["type"] == "button"
    assert stored_pattern["text"] == "Test Button"
    assert "visual_signature" in stored_pattern
    assert "id" in stored_pattern
    assert "timestamp" in stored_pattern
    assert stored_pattern["successful_interactions"] == 0


def test_find_similar_patterns(memory_system):
    """Test finding similar patterns based on visual signature"""
    # Create and store several patterns
    pattern1 = create_test_pattern()
    pattern2 = create_test_pattern()
    pattern2["type"] = "checkbox"
    pattern2["text"] = "Test Checkbox"
    
    # Add some noise to the second pattern's visual signature
    pattern2["visual_signature"] = pattern1["visual_signature"] * 0.9 + np.random.normal(0, 0.1, (64, 64))
    
    # Create a third pattern that's very different
    pattern3 = create_test_pattern()
    pattern3["type"] = "dropdown"
    pattern3["visual_signature"] = np.random.rand(64, 64)
    
    # Store patterns
    id1 = memory_system.store_pattern(pattern1)
    id2 = memory_system.store_pattern(pattern2)
    id3 = memory_system.store_pattern(pattern3)
    
    # Find similar patterns to the first one
    similar = memory_system.find_similar_patterns(pattern1["visual_signature"])
    
    # Should find at least the first and second patterns
    assert len(similar) >= 2
    
    # First one should be a perfect match
    assert similar[0]["similarity"] > 0.99
    
    # Check filtering by type
    button_matches = memory_system.find_similar_patterns(
        pattern1["visual_signature"],
        element_type="button"
    )
    assert len(button_matches) == 1
    assert button_matches[0]["type"] == "button"
    
    checkbox_matches = memory_system.find_similar_patterns(
        pattern1["visual_signature"],
        element_type="checkbox"
    )
    assert len(checkbox_matches) >= 1
    assert checkbox_matches[0]["type"] == "checkbox"


def test_record_interaction(memory_system):
    """Test recording interactions with UI elements"""
    # Create and store a pattern
    pattern = create_test_pattern()
    pattern_id = memory_system.store_pattern(pattern)
    
    # Record a successful interaction
    result = memory_system.record_interaction(
        pattern_id=pattern_id,
        action="click",
        success=True,
        context={"window_title": "Test Window"}
    )
    assert result is True
    
    # Record a failed interaction
    result = memory_system.record_interaction(
        pattern_id=pattern_id,
        action="click",
        success=False,
        context={"window_title": "Test Window"}
    )
    assert result is True
    
    # Check that the pattern's success/failure counts were updated
    stored_pattern = memory_system.get_pattern_by_id(pattern_id)
    assert stored_pattern["successful_interactions"] == 1
    assert stored_pattern["failed_interactions"] == 1
    
    # Check that the interaction history was updated
    assert len(memory_system.interaction_history) == 2
    
    # Try recording interaction with non-existent pattern
    result = memory_system.record_interaction(
        pattern_id="nonexistent_pattern",
        action="click",
        success=True
    )
    assert result is False


def test_enhance_detection(memory_system):
    """Test enhancing UI detection results with visual memory"""
    # Create and store patterns
    pattern = create_test_pattern()
    pattern_id = memory_system.store_pattern(pattern)
    
    # Record a successful interaction to influence confidence
    memory_system.record_interaction(
        pattern_id=pattern_id,
        action="click",
        success=True
    )
    
    # Create mock detection results
    ui_elements = [
        {
            "type": "button",
            "confidence": 0.7,
            "bbox": (50, 100, 150, 140),
            "center": (100, 120)
        }
    ]
    
    # Create a layout result with window context
    layout_results = {
        "window_title": "Test Window",
        "application": "TestApp"
    }
    
    # Create a mock screen image
    screen_image = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Create a patch for the _create_visual_signature method
    original_method = memory_system._create_visual_signature
    memory_system._create_visual_signature = lambda img: pattern["visual_signature"]
    
    # Enhance detection results
    enhanced = memory_system.enhance_detection(layout_results, ui_elements, screen_image)
    
    # Restore original method
    memory_system._create_visual_signature = original_method
    
    # Check that the detection was enhanced
    assert len(enhanced) == 1
    assert "memory_match" in enhanced[0]
    assert enhanced[0]["memory_match"]["pattern_id"] == pattern_id
    assert enhanced[0]["memory_match"]["similarity"] > 0.9
    assert enhanced[0]["memory_match"]["successful_interactions"] == 1


def test_similarity_calculation(memory_system):
    """Test calculation of visual similarity between signatures"""
    # Create two similar signatures
    sig1 = create_test_visual_signature()
    sig2 = sig1 * 0.9 + np.random.normal(0, 0.1, (64, 64))
    
    # Calculate similarity
    similarity = memory_system._calculate_similarity(sig1, sig2)
    
    # Should be high similarity (above 0.7)
    assert similarity > 0.7
    
    # Create a very different signature
    sig3 = np.random.rand(64, 64)
    
    # Calculate similarity
    similarity = memory_system._calculate_similarity(sig1, sig3)
    
    # Should be low similarity (below 0.7)
    assert similarity < 0.7
    
    # Test different shapes
    sig4 = np.random.rand(32, 32)
    similarity = memory_system._calculate_similarity(sig1, sig4)
    
    # Should still return a value
    assert 0 <= similarity <= 1


def test_visual_signature_creation(memory_system):
    """Test creation of visual signatures from images"""
    # Create a test image
    img = np.random.rand(100, 150, 3) * 255
    img = img.astype(np.uint8)
    
    # Create a visual signature
    signature = memory_system._create_visual_signature(img)
    
    # Check signature properties
    assert signature.shape == (64, 64)
    assert np.issubdtype(signature.dtype, np.floating)
    assert -10 < np.mean(signature) < 10  # Should be normalized
    
    # Test with grayscale image
    img_gray = np.random.rand(100, 150) * 255
    img_gray = img_gray.astype(np.uint8)
    
    signature = memory_system._create_visual_signature(img_gray)
    assert signature.shape == (64, 64)


def test_statistics(memory_system):
    """Test retrieving statistics about visual memory"""
    # Add several patterns
    pattern1 = create_test_pattern()
    pattern1["type"] = "button"
    
    pattern2 = create_test_pattern()
    pattern2["type"] = "checkbox"
    
    pattern3 = create_test_pattern()
    pattern3["type"] = "dropdown"
    
    # Store patterns
    id1 = memory_system.store_pattern(pattern1)
    id2 = memory_system.store_pattern(pattern2)
    id3 = memory_system.store_pattern(pattern3)
    
    # Record some interactions
    memory_system.record_interaction(id1, "click", True)
    memory_system.record_interaction(id1, "click", True)
    memory_system.record_interaction(id2, "click", False)
    
    # Get statistics
    stats = memory_system.get_statistics()
    
    # Check statistics properties
    assert stats["total_patterns"] == 3
    assert stats["successful_interactions"] == 2
    assert stats["failed_interactions"] == 1
    assert stats["overall_success_rate"] == 2/3
    
    # Check element type counts
    assert len(stats["element_types"]) == 3
    assert stats["element_types"]["button"] == 1
    assert stats["element_types"]["checkbox"] == 1
    assert stats["element_types"]["dropdown"] == 1


def test_memory_persistence(memory_system):
    """Test that visual memory is saved and loaded correctly"""
    # Add some patterns
    pattern1 = create_test_pattern()
    pattern2 = create_test_pattern()
    pattern2["type"] = "checkbox"
    
    # Store patterns
    id1 = memory_system.store_pattern(pattern1)
    id2 = memory_system.store_pattern(pattern2)
    
    # Record some interactions
    memory_system.record_interaction(id1, "click", True)
    
    # Save memory
    memory_system._save_memory()
    
    # Create a new memory system with the same path
    memory_path = memory_system.memory_path
    new_memory = VisualMemorySystem(config={"memory_path": memory_path})
    
    # Check that patterns were loaded
    assert len(new_memory.patterns) == 2
    assert new_memory.patterns[0]["id"] == id1
    assert new_memory.patterns[1]["id"] == id2
    
    # Check that interactions were loaded
    assert len(new_memory.interaction_history) == 1
    assert new_memory.interaction_history[0]["pattern_id"] == id1
    assert new_memory.interaction_history[0]["success"] is True


def test_memory_cleanup(memory_system):
    """Test cleaning up old or unused patterns"""
    # Add several patterns
    for i in range(10):
        pattern = create_test_pattern()
        pattern["type"] = f"element_{i}"
        memory_system.store_pattern(pattern)
    
    # Check initial count
    assert len(memory_system.patterns) == 10
    
    # Clean up to limit to 5 patterns
    removed = memory_system.clean_memory(max_patterns=5)
    
    # Should have removed 5 patterns
    assert removed == 5
    assert len(memory_system.patterns) == 5


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
