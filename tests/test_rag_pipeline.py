#!/usr/bin/env python
"""
Test script for NEXUS RAG Pipeline

This script tests the adaptive RAG pipeline to ensure it can:
1. Dynamically select different models based on query type
2. Fall back to alternative models when primary services are unavailable
3. Learn from successful and unsuccessful queries
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the parent directory to the path so we can import NEXUS modules
sys.path.append(str(Path(__file__).parent.parent))

# Import NEXUS modules
from src.ai_core.rag_engine.rag_pipeline import RAGPipeline
from src.integrations.groq_integration import GroqIntegration
from src.integrations.huggingface_integration import HuggingFaceIntegration
from src.integrations.ollama_integration import OllamaIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directory for test data if it doesn't exist
test_data_dir = Path("test_data")
test_data_dir.mkdir(exist_ok=True)


async def setup_integrations():
    """Set up all integrations needed for testing"""
    
    # Initialize integrations
    groq = GroqIntegration()
    huggingface = HuggingFaceIntegration()
    ollama = OllamaIntegration()
    
    # Check which integrations are available
    available_integrations = {}
    
    # For Groq - availability is determined by the 'available' flag
    if groq.available:
        available_integrations["groq"] = groq
        logger.info("✓ Groq integration initialized successfully")
    else:
        logger.warning("✗ Groq integration not available")
    
    # For Hugging Face - availability is determined by the 'available' flag
    if hasattr(huggingface, 'available') and huggingface.available:
        available_integrations["huggingface"] = huggingface
        logger.info("✓ Hugging Face integration initialized successfully")
    else:
        logger.warning("✗ Hugging Face integration not available")
    
    # For Ollama - availability is determined by the 'available' flag
    if hasattr(ollama, 'available') and ollama.available:
        available_integrations["ollama"] = ollama
        logger.info("✓ Ollama integration initialized successfully")
    else:
        logger.warning("✗ Ollama integration not available")
    
    return available_integrations


async def test_rag_pipeline(integrations):
    """Test the RAG Pipeline with various query types"""
    
    # Initialize RAG Pipeline
    rag = RAGPipeline(integrations=integrations)
    
    # Define test queries of different types
    test_queries = [
        {
            "type": "technical",
            "query": "Explain how async/await works in Python and give a code example."
        },
        {
            "type": "creative", 
            "query": "Write a short poem about artificial intelligence and learning."
        },
        {
            "type": "factual",
            "query": "What are the main components of a computer and their functions?"
        },
        {
            "type": "general",
            "query": "What are some effective ways to improve productivity?"
        }
    ]
    
    # Test results dictionary
    results = {}
    
    # Process each query and track which model was selected
    for test_case in test_queries:
        query_type = test_case["type"]
        query = test_case["query"]
        
        logger.info(f"Testing {query_type} query: {query}")
        
        # Process the query
        response = await rag.process_query(query)
        
        # Store results
        results[query_type] = {
            "query": query,
            "model_selected": response.get("model_used", "unknown"),
            "strategy_detected": response.get("strategy_detected", "unknown"),
            "success": response.get("success", False),
            "response_sample": response.get("response", "")[:100] + "..." if response.get("response") else "No response"
        }
        
        logger.info(f"Query processed with model: {response.get('model_used', 'unknown')}")
        logger.info(f"Strategy detected: {response.get('strategy_detected', 'unknown')}")
        logger.info(f"Success: {response.get('success', False)}")
        
    return results


async def test_rag_fallback(integrations):
    """Test the RAG Pipeline's fallback capabilities"""
    
    # Create a copy of integrations with some disabled
    limited_integrations = dict(integrations)
    
    # Disable the primary integration (assuming Groq is primary)
    if "groq" in limited_integrations:
        del limited_integrations["groq"]
        logger.info("Disabled Groq for fallback testing")
    
    # Initialize RAG Pipeline with limited integrations
    rag = RAGPipeline(integrations=limited_integrations)
    await rag.initialize()
    
    # Test query
    query = "Explain the concept of recursion in programming."
    
    logger.info(f"Testing fallback with query: {query}")
    
    # Process the query
    response = await rag.process_query(query)
    
    fallback_result = {
        "query": query,
        "model_selected": response.get("model_used", "unknown"),
        "success": response.get("success", False),
        "response_sample": response.get("response", "")[:100] + "..." if response.get("response") else "No response"
    }
    
    logger.info(f"Fallback test used model: {response.get('model_used', 'unknown')}")
    logger.info(f"Success: {response.get('success', False)}")
    
    return fallback_result


async def test_adaptive_learning(integrations):
    """Test if the RAG Pipeline learns from past queries"""
    
    # Initialize RAG Pipeline
    rag = RAGPipeline(integrations=integrations)
    await rag.initialize()
    
    # Run the same query multiple times to see if it improves
    repeated_query = "What is the best way to structure a Python project?"
    
    learning_results = []
    
    # Run the query 3 times
    for i in range(3):
        logger.info(f"Adaptive learning test - Run {i+1} with query: {repeated_query}")
        
        # Process the query
        start_time = asyncio.get_event_loop().time()
        response = await rag.process_query(repeated_query)
        end_time = asyncio.get_event_loop().time()
        
        # Store results
        run_result = {
            "run": i+1,
            "model_selected": response.get("model_used", "unknown"),
            "strategy_detected": response.get("strategy_detected", "unknown"),
            "response_time": end_time - start_time,
            "success": response.get("success", False)
        }
        
        learning_results.append(run_result)
        logger.info(f"Run {i+1} processed with model: {response.get('model_used', 'unknown')}")
        logger.info(f"Response time: {run_result['response_time']:.2f} seconds")
        
    return learning_results


async def main():
    """Main test function"""
    logger.info("Starting NEXUS RAG Pipeline Tests")
    
    # Setup integrations
    integrations = await setup_integrations()
    
    if not integrations:
        logger.error("No integrations available. Cannot proceed with tests.")
        return
    
    logger.info(f"Available integrations: {', '.join(integrations.keys())}")
    
    # Run tests
    try:
        # Test 1: RAG Pipeline with different query types
        logger.info("=== Test 1: RAG Pipeline with Different Query Types ===")
        rag_results = await test_rag_pipeline(integrations)
        
        # Test 2: RAG Fallback capabilities
        logger.info("=== Test 2: RAG Fallback Capabilities ===")
        fallback_result = await test_rag_fallback(integrations)
        
        # Test 3: Adaptive learning
        logger.info("=== Test 3: Adaptive Learning ===")
        learning_results = await test_adaptive_learning(integrations)
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        logger.info("RAG Pipeline Tests:")
        for query_type, result in rag_results.items():
            logger.info(f"  - {query_type}: Used {result['model_selected']} ({result['strategy_detected']})")
        
        logger.info(f"Fallback Test: Used {fallback_result['model_selected']}")
        
        logger.info("Adaptive Learning Test:")
        for run in learning_results:
            logger.info(f"  - Run {run['run']}: {run['model_selected']} in {run['response_time']:.2f}s")
            
        # Check for improvements in adaptive learning
        if learning_results:
            first_run = learning_results[0]['response_time']
            last_run = learning_results[-1]['response_time']
            
            if last_run < first_run:
                logger.info(f"✓ Adaptive learning improved response time by {(first_run - last_run):.2f}s")
            else:
                logger.info(f"✗ No improvement in response time observed")
                
    except Exception as e:
        logger.error(f"Error during testing: {e}")
    finally:
        # Clean up
        for integration in integrations.values():
            if hasattr(integration, 'close'):
                await integration.close()
        
        logger.info("NEXUS RAG Pipeline Tests Completed")


if __name__ == "__main__":
    asyncio.run(main())
