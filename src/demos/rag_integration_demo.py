"""
NEXUS RAG Integration Demo
Demonstrates how the NEXUS architecture dynamically selects and combines AI tools
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from integrations.integration_manager import IntegrationManager
from integrations.groq_integration import GroqIntegration
from integrations.huggingface_integration import HuggingFaceIntegration
from integrations.gemini_integration import GeminiAI
from ai_core.rag_engine.vector_storage import VectorStorage
from ai_core.rag_engine.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the RAG integration demo"""
    print("\n" + "="*80)
    print("NEXUS RAG Integration Demo")
    print("="*80 + "\n")
    
    # Initialize the integration manager
    print("Initializing NEXUS Integration Manager...")
    integration_manager = IntegrationManager()
    
    # Manually register our LLM services (in case auto-discovery didn't work)
    print("Registering AI services...")
    
    # Register Groq
    groq_integration = GroqIntegration()
    integration_manager.register_custom_integration("groq", groq_integration)
    
    # Register Hugging Face
    hf_integration = HuggingFaceIntegration()
    integration_manager.register_custom_integration("huggingface", hf_integration)
    
    # Register Gemini if available
    try:
        gemini_integration = GeminiAI()
        integration_manager.register_custom_integration("gemini", gemini_integration)
    except ImportError:
        print("Gemini integration not available")
    
    # Initialize vector storage
    print("Initializing Vector Storage...")
    vector_storage = VectorStorage(storage_dir="memory/demo_vector_db")
    
    # Add some sample knowledge to the vector storage
    print("Adding sample knowledge to vector storage...")
    
    sample_texts = [
        "NEXUS is an AI orchestration system that dynamically selects and combines tools.",
        "The RAG (Retrieval-Augmented Generation) paradigm enhances LLM responses with relevant knowledge.",
        "Hugging Face provides access to thousands of AI models for various tasks.",
        "Groq is known for its fast inference speeds for large language models.",
        "Python is a popular programming language for AI and machine learning.",
        "The transformer architecture revolutionized natural language processing."
    ]
    
    sample_metadata = [
        {"source": "NEXUS Documentation", "topic": "architecture"},
        {"source": "AI Research Paper", "topic": "rag"},
        {"source": "API Documentation", "topic": "huggingface"},
        {"source": "API Documentation", "topic": "groq"},
        {"source": "Programming Guide", "topic": "python"},
        {"source": "AI Research Paper", "topic": "transformers"}
    ]
    
    vector_storage.add_texts(sample_texts, sample_metadata)
    
    # Create a RAG pipeline with dynamic LLM selection
    print("Creating RAG pipeline...")
    
    # Function to select the best LLM based on task
    async def dynamic_llm_selector(task_type):
        """Dynamically select the best LLM for a task"""
        best_provider = await integration_manager.get_best_provider("text_generation", {"task_type": task_type})
        if best_provider:
            return integration_manager.integrations[best_provider]
        return None
    
    # Initialize the RAG pipeline with dynamic LLM selection
    pipeline = RAGPipeline(
        vector_storage=vector_storage,
        llm_service=await dynamic_llm_selector("general")
    )
    
    # Demo the RAG pipeline
    print("\nDemonstrating RAG pipeline with different queries...\n")
    
    demo_queries = [
        {"query": "What is NEXUS?", "task_type": "general"},
        {"query": "Explain how RAG works", "task_type": "technical"},
        {"query": "What are some fast LLM options?", "task_type": "factual"}
    ]
    
    for demo in demo_queries:
        query = demo["query"]
        task_type = demo["task_type"]
        
        print(f"\n>>> Query: {query} (Task type: {task_type})")
        print("-" * 50)
        
        # Dynamically select LLM based on task type
        pipeline.llm_service = await dynamic_llm_selector(task_type)
        
        # Process the query
        result = await pipeline.query(query, task_type=task_type)
        
        print(f"Response:")
        print(result.get("response", "No response generated"))
        print(f"\nUsed model: {result.get('model_used', 'Unknown')}")
        print(f"Elapsed time: {result.get('elapsed_time', 0):.2f} seconds")
        print("-" * 50)
    
    # Demo dynamic tool selection
    print("\n\nDemonstrating dynamic tool selection for different capabilities...\n")
    
    capabilities = ["text_generation", "image_analysis", "text_embedding"]
    
    for capability in capabilities:
        print(f">>> Looking for best provider for: {capability}")
        provider = await integration_manager.get_best_provider(capability)
        
        if provider:
            print(f"Selected provider: {provider}")
            
            if capability == "text_generation":
                # Demo text generation with the selected provider
                result = await integration_manager.execute_capability(
                    "text_generation",
                    provider=provider,
                    prompt="Explain the concept of AI orchestration in one paragraph."
                )
                
                if isinstance(result, dict) and "text" in result:
                    print(f"\nGenerated text: {result['text'][:200]}...")
                else:
                    print(f"\nResult: {result}")
        else:
            print(f"No provider available for {capability}")
        
        print("-" * 50)
    
    print("\nDemo completed!")
    
    # Print discovered capabilities
    print("\nDiscovered capabilities in the system:")
    for capability, providers in integration_manager.capability_providers.items():
        print(f"  - {capability}: {', '.join(providers)}")

if __name__ == "__main__":
    asyncio.run(main())
