#!/usr/bin/env python
"""
Test script for NEXUS Shared Memory System

This script tests the shared memory system to ensure it can:
1. Store and retrieve experiences
2. Learn from successes and failures
3. Provide relevant context for future tasks
4. Enable the system to adapt instead of following rigid rules
"""

import os
import sys
import asyncio
import logging
import random
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import NEXUS modules
sys.path.append(str(Path(__file__).parent.parent))

# Import NEXUS modules
from src.ai_core.shared_memory import SharedMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directory for test data if it doesn't exist
test_data_dir = Path("test_data")
test_data_dir.mkdir(exist_ok=True)
memory_dir = test_data_dir / "memory"
memory_dir.mkdir(exist_ok=True)


def generate_test_experiences():
    """Generate a set of test experiences for the memory system"""
    
    domains = ["web_search", "image_analysis", "text_generation", "system_automation"]
    actions = ["search", "analyze", "generate", "automate", "extract", "process"]
    statuses = [True, False]  # Success or failure
    
    experiences = []
    
    # Generate 15 random experiences
    for i in range(15):
        domain = random.choice(domains)
        action = random.choice(actions)
        success = random.choice(statuses)
        
        # Weight toward successful experiences (70% success)
        if i % 10 < 7:
            success = True
        
        # Create experience
        experience = {
            "id": f"exp_{i+1}",
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "description": f"{action.capitalize()} {domain.replace('_', ' ')} task",
            "success": success,
            "details": {
                "action": action,
                "parameters": {
                    "complexity": random.randint(1, 5),
                    "duration": random.uniform(0.5, 5.0)
                },
                "outcome": "completed successfully" if success else "failed"
            },
            "learning_points": [
                f"{'Effective' if success else 'Ineffective'} approach for {domain} tasks",
                f"{'Success' if success else 'Failure'} pattern identified for {action} operations"
            ]
        }
        
        experiences.append(experience)
    
    return experiences


def test_basic_memory_operations():
    """Test basic memory operations (store, retrieve, update)"""
    
    # Initialize shared memory
    logger.info("Initializing Shared Memory...")
    memory = SharedMemory(storage_dir=str(memory_dir))
    # Removed the call to initialize() method
    
    # Test context operations
    logger.info("Testing context operations...")
    
    # Create session context
    memory.create_session_context("Test basic memory operations")
    
    # Update context
    memory.update_context("test_key", "test_value")
    memory.update_context("test_dict", {"a": 1, "b": 2})
    memory.update_context("test_list", [1, 2, 3])
    
    # Get context
    context_value = memory.get_context("test_key")
    context_dict = memory.get_context("test_dict")
    context_list = memory.get_context("test_list")
    
    # Verify context operations
    context_ops_success = (
        context_value == "test_value" and
        isinstance(context_dict, dict) and
        isinstance(context_list, list)
    )
    
    if context_ops_success:
        logger.info("✓ Basic context operations successful")
    else:
        logger.warning("✗ Basic context operations failed")
    
    # Test preference operations
    logger.info("Testing preference operations...")
    
    # Store preferences
    memory.store_user_preference("color_theme", "dark")
    memory.store_user_preference("response_style", "concise")
    
    # Get preferences
    theme_pref = memory.get_user_preference("color_theme")
    style_pref = memory.get_user_preference("response_style")
    
    # Verify preference operations
    pref_ops_success = (
        theme_pref == "dark" and
        style_pref == "concise"
    )
    
    if pref_ops_success:
        logger.info("✓ Preference operations successful")
    else:
        logger.warning("✗ Preference operations failed")
    
    return {
        "context_operations_success": context_ops_success,
        "preference_operations_success": pref_ops_success
    }


def test_experience_learning():
    """Test the experience learning capabilities"""
    
    # Initialize shared memory
    logger.info("Testing experience learning...")
    memory = SharedMemory(storage_dir=str(memory_dir))
    
    # Generate test experiences
    experiences = generate_test_experiences()
    logger.info(f"Generated {len(experiences)} test experiences")
    
    # Store experiences in memory
    for exp in experiences:
        if exp["success"]:
            memory.learn_from_success({
                "id": exp["id"],
                "description": exp["description"],
                "domain": exp["domain"],
                "details": exp["details"],
                "learning_points": exp["learning_points"]
            })
        else:
            memory.learn_from_error({
                "id": exp["id"],
                "description": exp["description"],
                "domain": exp["domain"],
                "details": exp["details"],
                "learning_points": exp["learning_points"],
                "error_type": "task_failure"
            })
    
    # Count stored experiences
    success_count = len(memory.get_success_patterns())
    error_count = len(memory.get_error_patterns())
    
    logger.info(f"Stored {success_count} success patterns and {error_count} error patterns")
    
    # Verify counts match expected
    expected_success = sum(1 for exp in experiences if exp["success"])
    expected_errors = sum(1 for exp in experiences if not exp["success"])
    
    counts_match = (
        success_count == expected_success and
        error_count == expected_errors
    )
    
    if counts_match:
        logger.info("✓ Experience count verification successful")
    else:
        logger.warning(f"✗ Experience count verification failed. Expected {expected_success} successes and {expected_errors} errors.")
    
    return {
        "success_count": success_count,
        "error_count": error_count,
        "counts_match": counts_match
    }


def test_memory_relevance():
    """Test if the memory system returns relevant experiences"""
    
    # Initialize shared memory
    logger.info("Testing memory relevance...")
    memory = SharedMemory(storage_dir=str(memory_dir))
    
    # Define test queries
    test_queries = [
        "Search for information about Python programming",
        "Analyze this image and tell me what's in it",
        "Generate a creative story about AI",
        "Automate opening the web browser"
    ]
    
    relevance_results = {}
    
    # Test each query for relevant experiences
    for query in test_queries:
        logger.info(f"Testing relevance for query: {query}")
        
        # Get related experiences
        related = memory.get_related_experiences(query)
        
        # Count experiences
        count = len(related)
        
        # Check if any are returned
        has_related = count > 0
        
        # Log results
        if has_related:
            logger.info(f"✓ Found {count} related experiences for '{query}'")
            for i, exp in enumerate(related[:2]):
                logger.info(f"  - Related {i+1}: {exp.get('description', 'Unknown')}")
        else:
            logger.warning(f"✗ No related experiences found for '{query}'")
        
        # Store results
        relevance_results[query] = {
            "count": count,
            "has_related": has_related
        }
    
    return relevance_results


def test_adaptive_learning():
    """Test if the memory system helps adaptation over repeated tasks"""
    
    # Initialize shared memory
    logger.info("Testing adaptive learning...")
    memory = SharedMemory(storage_dir=str(memory_dir))
    
    # Create a sample task domain
    task_domain = "image_analysis"
    task_description = "Analyze an image of a landscape"
    
    # Create failure experiences first, then successes to simulate learning
    failure_experiences = [
        {
            "id": "adapt_1",
            "description": task_description,
            "domain": task_domain,
            "success": False,
            "details": {
                "method": "basic_vision",
                "parameters": {"model": "simple"},
                "error": "insufficient_detail"
            },
            "learning_points": ["Basic vision is not sufficient for landscape analysis"]
        },
        {
            "id": "adapt_2",
            "description": task_description,
            "domain": task_domain,
            "success": False,
            "details": {
                "method": "cloud_vision",
                "parameters": {"quality": "low"},
                "error": "api_timeout"
            },
            "learning_points": ["Low quality settings cause timeouts"]
        }
    ]
    
    success_experiences = [
        {
            "id": "adapt_3",
            "description": task_description,
            "domain": task_domain,
            "success": True,
            "details": {
                "method": "enhanced_vision",
                "parameters": {"quality": "high", "model": "landscape_specialist"},
                "outcome": "detailed_analysis"
            },
            "learning_points": ["Specialist models work better for specific domains"]
        },
        {
            "id": "adapt_4",
            "description": task_description,
            "domain": task_domain,
            "success": True,
            "details": {
                "method": "adaptive_vision",
                "parameters": {"adaptive": True, "learn_from_past": True},
                "outcome": "comprehensive_analysis"
            },
            "learning_points": ["Adaptive approaches with learning are most effective"]
        }
    ]
    
    # Store the experiences in chronological order
    for exp in failure_experiences:
        memory.learn_from_error({
            "id": exp["id"],
            "description": exp["description"],
            "domain": exp["domain"],
            "details": exp["details"],
            "learning_points": exp["learning_points"],
            "error_type": "task_failure"
        })
    
    for exp in success_experiences:
        memory.learn_from_success({
            "id": exp["id"],
            "description": exp["description"],
            "domain": exp["domain"],
            "details": exp["details"],
            "learning_points": exp["learning_points"]
        })
    
    # Now query for similar task and check if it recommends the most successful approach
    test_task = "Analyze an image of mountains and forests"
    related = memory.get_related_experiences(test_task)
    
    # We want the most successful and recent experiences first
    adaptation_working = False
    recommended_approach = None
    
    if related and len(related) > 0:
        # The most relevant experience should be one of our success stories
        top_experience = related[0]
        top_id = top_experience.get("id", "")
        
        # Check if it's one of our success experiences
        success_ids = [exp["id"] for exp in success_experiences]
        adaptation_working = top_id in success_ids
        
        if adaptation_working:
            recommended_approach = top_experience.get("details", {}).get("method", "unknown")
            logger.info(f"✓ Adaptation working. Recommended approach: {recommended_approach}")
        else:
            logger.warning("✗ Adaptation not working as expected. Top experience not from success patterns.")
    else:
        logger.warning("✗ No related experiences found for adaptive learning test")
    
    return {
        "adaptation_working": adaptation_working,
        "recommended_approach": recommended_approach
    }


def test_persistence():
    """Test if the memory system persists data between instances"""
    
    # First instance: store data
    logger.info("Testing memory persistence...")
    
    # Create unique test key
    test_key = f"persistence_test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    test_value = f"test_value_{random.randint(1000, 9999)}"
    
    # First instance
    logger.info(f"First instance: storing {test_key}={test_value}")
    memory1 = SharedMemory(storage_dir=str(memory_dir))
    memory1.store_user_preference(test_key, test_value)
    
    # Create a second instance
    logger.info("Second instance: checking for persistence")
    memory2 = SharedMemory(storage_dir=str(memory_dir))
    retrieved_value = memory2.get_user_preference(test_key)
    
    # Check if values match
    persistence_working = retrieved_value == test_value
    
    if persistence_working:
        logger.info(f"✓ Persistence working. Retrieved {test_key}={retrieved_value}")
    else:
        logger.warning(f"✗ Persistence not working. Expected {test_value}, got {retrieved_value}")
    
    return {
        "persistence_working": persistence_working,
        "test_key": test_key,
        "expected_value": test_value,
        "retrieved_value": retrieved_value
    }


def main():
    """Main test function"""
    logger.info("Starting NEXUS Shared Memory System Tests")
    
    try:
        # Test 1: Basic Memory Operations
        logger.info("=== Test 1: Basic Memory Operations ===")
        basic_results = test_basic_memory_operations()
        
        # Test 2: Experience Learning
        logger.info("=== Test 2: Experience Learning ===")
        experience_results = test_experience_learning()
        
        # Test 3: Memory Relevance
        logger.info("=== Test 3: Memory Relevance ===")
        relevance_results = test_memory_relevance()
        
        # Test 4: Adaptive Learning
        logger.info("=== Test 4: Adaptive Learning ===")
        adaptive_results = test_adaptive_learning()
        
        # Test 5: Persistence
        logger.info("=== Test 5: Persistence ===")
        persistence_results = test_persistence()
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        
        # Basic operations
        logger.info("Basic Memory Operations:")
        logger.info(f"  Context operations: {'✓' if basic_results['context_operations_success'] else '✗'}")
        logger.info(f"  Preference operations: {'✓' if basic_results['preference_operations_success'] else '✗'}")
        
        # Experience learning
        logger.info("Experience Learning:")
        logger.info(f"  Stored {experience_results['success_count']} success patterns")
        logger.info(f"  Stored {experience_results['error_count']} error patterns")
        logger.info(f"  Counts match expected: {'✓' if experience_results['counts_match'] else '✗'}")
        
        # Memory relevance
        logger.info("Memory Relevance:")
        found_count = sum(1 for result in relevance_results.values() if result["has_related"])
        logger.info(f"  Found relevant experiences for {found_count}/{len(relevance_results)} queries")
        
        # Adaptive learning
        logger.info("Adaptive Learning:")
        if adaptive_results["adaptation_working"]:
            logger.info(f"  ✓ Successfully recommended {adaptive_results['recommended_approach']}")
        else:
            logger.info("  ✗ Failed to recommend adaptive approach")
        
        # Persistence
        logger.info("Persistence:")
        logger.info(f"  {'✓' if persistence_results['persistence_working'] else '✗'} Memory persists between instances")
        
        # Overall assessment
        all_tests_passed = (
            basic_results["context_operations_success"] and
            basic_results["preference_operations_success"] and
            experience_results["counts_match"] and
            found_count > 0 and
            adaptive_results["adaptation_working"] and
            persistence_results["persistence_working"]
        )
        
        if all_tests_passed:
            logger.info("\n✓ All Shared Memory tests PASSED")
        else:
            logger.info("\n✗ Some Shared Memory tests FAILED")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    
    logger.info("NEXUS Shared Memory System Tests Completed")


if __name__ == "__main__":
    main()
