"""
Unit tests for clarification engine module

These tests verify that the clarification engine correctly generates questions,
processes user responses, and learns from interactions to improve future automation.
"""
import os
import time
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ai_core.automation.clarification_engine import ClarificationEngine

@pytest.fixture
def clarification_engine():
    """Create a ClarificationEngine instance for testing"""
    # Create with default settings
    engine = ClarificationEngine()
    
    # Set up a mock response callback
    mock_callback = MagicMock(return_value="yes")
    engine.set_response_callback(mock_callback)
    
    yield engine


@pytest.fixture
def custom_clarification_engine():
    """Create a ClarificationEngine with custom configuration"""
    # Create temporary directory for memory
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Create with custom settings
    config = {
        "confidence_threshold": 0.8,
        "max_attempts": 2,
        "learning_rate": 0.3,
        "memory_path": temp_dir,
        "enable_learning": True,
        "question_templates": {
            "custom_scenario": [
                "Custom question template with {placeholder}?"
            ]
        }
    }
    
    engine = ClarificationEngine(config=config)
    
    # Set up a mock response callback
    mock_callback = MagicMock(return_value="yes")
    engine.set_response_callback(mock_callback)
    
    yield engine
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def test_initialization():
    """Test clarification engine initialization"""
    # Default configuration
    engine = ClarificationEngine()
    assert engine.confidence_threshold == 0.7
    assert engine.max_attempts == 3
    assert engine.enable_learning is True
    
    # Custom configuration
    config = {
        "confidence_threshold": 0.8,
        "max_attempts": 2,
        "learning_rate": 0.3,
        "enable_learning": False
    }
    engine = ClarificationEngine(config=config)
    assert engine.confidence_threshold == 0.8
    assert engine.max_attempts == 2
    assert engine.learning_rate == 0.3
    assert engine.enable_learning is False


def test_needs_clarification(clarification_engine):
    """Test determining if clarification is needed"""
    # Should need clarification
    assert clarification_engine.needs_clarification(0.5) is True
    
    # Should not need clarification
    assert clarification_engine.needs_clarification(0.8) is False
    
    # Edge case - exactly at threshold
    assert clarification_engine.needs_clarification(0.7) is False


def test_generate_question(clarification_engine):
    """Test generating clarification questions"""
    # Test UI element action scenario
    context = {
        "element_type": "button",
        "element_text": "Submit",
        "action": "click"
    }
    question = clarification_engine.generate_question("ui_element_action", context)
    assert "button" in question
    assert "Submit" in question
    assert "click" in question
    
    # Test ambiguous UI element scenario
    context = {
        "element_type": "button",
        "options": "Submit, Cancel, Help",
        "count": 3
    }
    question = clarification_engine.generate_question("ui_element_ambiguous", context)
    assert "button" in question
    assert "Submit, Cancel, Help" in question
    
    # Test dangerous action scenario
    context = {
        "action": "delete all files"
    }
    question = clarification_engine.generate_question("dangerous_action", context)
    assert "delete all files" in question
    assert "risky" in question.lower() or "dangerous" in question.lower()
    
    # Test fallback to general uncertainty for unknown scenario
    context = {
        "action": "unknown action"
    }
    question = clarification_engine.generate_question("unknown_scenario", context)
    assert "unknown action" in question
    
    # Test missing context keys
    question = clarification_engine.generate_question("ui_element_action", {"action": "click"})
    assert "click" in question
    assert "proceed" in question.lower()


def test_process_response(clarification_engine):
    """Test processing user responses"""
    # Test positive responses
    assert clarification_engine._process_response("Yes", 0.5) == (True, 0.8)
    assert clarification_engine._process_response("sure, go ahead", 0.5) == (True, 0.8)
    assert clarification_engine._process_response("yes, that's correct", 0.5) == (True, 0.8)
    
    # Test negative responses
    assert clarification_engine._process_response("No", 0.5) == (False, 0.4)
    assert clarification_engine._process_response("stop, don't do that", 0.5) == (False, 0.4)
    assert clarification_engine._process_response("that's wrong", 0.5) == (False, 0.4)
    
    # Test unclear responses (should slightly favor proceeding)
    assert clarification_engine._process_response("hmm, not sure", 0.5) == (True, 0.55)
    assert clarification_engine._process_response("I guess", 0.5) == (True, 0.55)
    
    # Test confidence capping
    assert clarification_engine._process_response("Yes", 0.9) == (True, 1.0)  # Capped at 1.0
    assert clarification_engine._process_response("No", 0.1) == (False, 0.0)  # Capped at 0.0


def test_ask_for_clarification(clarification_engine):
    """Test asking for clarification"""
    # Setup mock response
    mock_callback = MagicMock(return_value="Yes, that's correct")
    clarification_engine.set_response_callback(mock_callback)
    
    # Test UI element action scenario
    context = {
        "element_type": "button",
        "element_text": "Submit",
        "action": "click"
    }
    result = clarification_engine.ask_for_clarification("ui_element_action", context, 0.5)
    
    # Check result
    assert result["proceed"] is True
    assert result["confidence"] > 0.5
    assert "button" in result["question"]
    assert "Submit" in result["question"]
    assert "clarification_id" in result
    
    # Check that callback was called with the question
    mock_callback.assert_called_once()
    
    # Check that history was updated
    assert len(clarification_engine.clarification_history) == 1
    assert clarification_engine.clarification_history[0]["scenario"] == "ui_element_action"
    
    # Test with no callback set
    clarification_engine._response_callback = None
    result = clarification_engine.ask_for_clarification("ui_element_action", context, 0.5)
    assert result["proceed"] is False
    assert result["question"] is None


def test_similar_clarifications(clarification_engine):
    """Test finding similar clarification scenarios"""
    # Add some history
    clarification_engine.clarification_history = [
        {
            "scenario": "ui_element_action",
            "context": {
                "element_type": "button",
                "element_text": "Submit",
                "action": "click"
            },
            "question": "Should I click the Submit button?",
            "response": "Yes",
            "proceed": True
        },
        {
            "scenario": "ui_element_action",
            "context": {
                "element_type": "button",
                "element_text": "Cancel",
                "action": "click"
            },
            "question": "Should I click the Cancel button?",
            "response": "No",
            "proceed": False
        },
        {
            "scenario": "dangerous_action",
            "context": {
                "action": "delete file"
            },
            "question": "Should I delete the file?",
            "response": "Yes",
            "proceed": True
        }
    ]
    
    # Find similar to first one
    context = {
        "element_type": "button",
        "element_text": "Submit Form",
        "action": "click"
    }
    similar = clarification_engine.get_similar_clarifications(context)
    
    # Should find similar ones
    assert len(similar) > 0
    assert similar[0]["context"]["element_type"] == "button"
    assert "Submit" in similar[0]["context"]["element_text"]
    
    # Find similar to dangerous action
    context = {
        "action": "delete files"
    }
    similar = clarification_engine.get_similar_clarifications(context)
    
    # Should find the dangerous action
    assert len(similar) > 0
    assert similar[0]["context"]["action"] == "delete file"


def test_context_similarity(clarification_engine):
    """Test calculating context similarity"""
    # Exact match
    context1 = {
        "element_type": "button",
        "action": "click",
        "element_text": "Submit"
    }
    context2 = {
        "element_type": "button",
        "action": "click",
        "element_text": "Submit"
    }
    similarity = clarification_engine._calculate_context_similarity(context1, context2)
    assert similarity > 0.9
    
    # Partial match
    context2 = {
        "element_type": "button",
        "action": "hover",
        "element_text": "Submit"
    }
    similarity = clarification_engine._calculate_context_similarity(context1, context2)
    assert 0.5 < similarity < 0.9
    
    # Poor match
    context2 = {
        "element_type": "textbox",
        "action": "type",
        "element_text": "Username"
    }
    similarity = clarification_engine._calculate_context_similarity(context1, context2)
    assert similarity < 0.5
    
    # Text similarity
    context1 = {
        "element_text": "Please submit your form"
    }
    context2 = {
        "element_text": "Submit form now"
    }
    similarity = clarification_engine._calculate_context_similarity(context1, context2)
    assert similarity > 0.0  # Should find some similarity due to "submit" and "form"


def test_learning_from_clarification(custom_clarification_engine):
    """Test learning from clarification"""
    # Create several clarifications with positive responses
    for i in range(10):
        custom_clarification_engine.clarification_history.append({
            "scenario": "ui_element_action",
            "context": {
                "element_type": "button",
                "element_text": f"Button {i}",
                "action": "click"
            },
            "question": f"Should I click Button {i}?",
            "response": "Yes",
            "proceed": True
        })
    
    # Store original threshold
    original_threshold = custom_clarification_engine.confidence_threshold
    
    # Process a new clarification (should learn from history)
    custom_clarification_engine._learn_from_clarification({
        "scenario": "ui_element_action",
        "context": {
            "element_type": "button",
            "element_text": "New Button",
            "action": "click"
        },
        "question": "Should I click New Button?",
        "response": "Yes",
        "proceed": True
    })
    
    # Since we've had consistently positive responses, threshold should decrease
    assert custom_clarification_engine.confidence_threshold < original_threshold


def test_statistics(clarification_engine):
    """Test retrieving clarification statistics"""
    # Add some history
    clarification_engine.clarification_history = [
        {
            "scenario": "ui_element_action",
            "proceed": True
        },
        {
            "scenario": "ui_element_action",
            "proceed": True
        },
        {
            "scenario": "dangerous_action",
            "proceed": False
        }
    ]
    
    # Get statistics
    stats = clarification_engine.get_clarification_statistics()
    
    # Check statistics
    assert stats["total_clarifications"] == 3
    assert stats["proceed_rate"] == 2/3
    assert stats["current_confidence_threshold"] == clarification_engine.confidence_threshold
    assert "most_common_scenarios" in stats
    assert stats["most_common_scenarios"]["ui_element_action"] == 2
    assert stats["most_common_scenarios"]["dangerous_action"] == 1


def test_memory_persistence(custom_clarification_engine):
    """Test saving and loading clarification history"""
    # Create some history
    custom_clarification_engine.clarification_history = [
        {
            "id": "test_1",
            "scenario": "ui_element_action",
            "context": {
                "element_type": "button",
                "element_text": "Submit",
                "action": "click"
            },
            "timestamp": time.time(),
            "question": "Should I click Submit?",
            "response": "Yes",
            "proceed": True,
            "initial_confidence": 0.5,
            "updated_confidence": 0.8
        }
    ]
    
    # Save history
    custom_clarification_engine._save_history()
    
    # Clear history
    custom_clarification_engine.clarification_history = []
    
    # Load history
    custom_clarification_engine._load_history()
    
    # Check that history was loaded
    assert len(custom_clarification_engine.clarification_history) == 1
    assert custom_clarification_engine.clarification_history[0]["id"] == "test_1"
    assert custom_clarification_engine.clarification_history[0]["scenario"] == "ui_element_action"


def test_custom_templates(custom_clarification_engine):
    """Test custom question templates"""
    # Test custom template
    context = {
        "placeholder": "test value"
    }
    question = custom_clarification_engine.generate_question("custom_scenario", context)
    assert question == "Custom question template with test value?"
    
    # Add new templates
    custom_clarification_engine.add_question_templates("new_scenario", [
        "New template with {placeholder}?",
        "Alternative template with {placeholder}?"
    ])
    
    # Test new template
    question = custom_clarification_engine.generate_question("new_scenario", context)
    assert "test value" in question
    
    # Test template rotation
    custom_clarification_engine.clarification_history = [{}]  # Add one entry to shift index
    question = custom_clarification_engine.generate_question("new_scenario", context)
    assert "Alternative template" in question


def test_example_clarification(clarification_engine):
    """Test generating example clarifications"""
    # Get example for UI action
    example = clarification_engine.get_example_clarification("ui_element_action")
    
    # Check example
    assert example["scenario"] == "ui_element_action"
    assert "element_type" in example["context"]
    assert "element_text" in example["context"]
    assert "action" in example["context"]
    assert example["example"] is True
    assert example["question"] is not None
    
    # Get example for unknown scenario (should use generic context)
    example = clarification_engine.get_example_clarification("unknown_scenario")
    
    # Should still have a question
    assert example["question"] is not None


def test_no_callback_behavior(clarification_engine):
    """Test behavior when no response callback is set"""
    # Clear callback
    clarification_engine._response_callback = None
    
    # Try to ask for clarification
    result = clarification_engine.ask_for_clarification("ui_element_action", {
        "element_type": "button",
        "element_text": "Submit",
        "action": "click"
    })
    
    # Should return failed result
    assert result["proceed"] is False
    assert result["question"] is None
    assert result["response"] is None


def test_adaptive_questioning(clarification_engine):
    """Test adaptive questioning based on history"""
    # Add some history to rotate questions
    for i in range(5):
        clarification_engine.clarification_history.append({
            "scenario": "ui_element_action",
            "context": {
                "element_type": "button",
                "element_text": f"Button {i}",
                "action": "click"
            }
        })
    
    # Generate multiple questions for the same scenario
    context = {
        "element_type": "button",
        "element_text": "Submit",
        "action": "click"
    }
    
    # Get questions
    questions = []
    for i in range(3):
        # Add history entry to rotate template
        clarification_engine.clarification_history.append({})
        
        # Generate question
        question = clarification_engine.generate_question("ui_element_action", context)
        questions.append(question)
    
    # Should have different question formats
    assert len(set(questions)) > 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
