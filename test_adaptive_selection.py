"""
Demonstrate NEXUS's adaptive model selection capabilities
"""
import asyncio
import logging
import time
import random
from src.ai_core.rag_engine.rag_pipeline import RAGPipeline
from src.ai_core.rag_engine.vector_storage import VectorStorage
from src.integrations.ollama_integration import OllamaIntegration
from src.integrations.groq_integration import GroqIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMService:
    """Mock LLM service that simulates different response times and success rates"""
    
    def __init__(self, name, success_rate=0.9, avg_time=1.0):
        self.name = name
        self.success_rate = success_rate
        self.avg_time = avg_time
        self.available = True
        
    async def generate_text(self, prompt, max_tokens=1000):
        # Simulate processing time
        await asyncio.sleep(self.avg_time * random.uniform(0.7, 1.3))
        
        # Simulate success/failure
        if random.random() <= self.success_rate:
            return {"text": f"Response from {self.name}: Successfully processed the prompt."}
        else:
            return {"text": f"Error from {self.name}", "error": True}
    
    async def list_models(self):
        if self.name == "ollama":
            return [{"id": "deepseek-coder:latest"}, {"id": "llama2:latest"}]
        elif self.name == "groq":
            return [{"id": "mixtral-8x7b-32768"}, {"id": "llama2-70b-4096"}]
        return []
    
    async def get_available_model(self):
        if self.name == "ollama":
            return "ollama:deepseek-coder"
        elif self.name == "groq":
            return "groq:mixtral-8x7b-32768"
        return None
    
    async def is_model_available(self, model_name):
        return True

async def test_adaptive_selection():
    """Test the adaptive model selection capabilities of NEXUS"""
    logger.info("Starting adaptive model selection test")
    
    # Create mock LLM services with different characteristics
    # Model A: Fast but less reliable
    model_a = MockLLMService("ollama", success_rate=0.7, avg_time=0.5)
    # Model B: Slower but very reliable
    model_b = MockLLMService("groq", success_rate=0.95, avg_time=1.2)
    
    # Create a simple vector storage
    vector_storage = VectorStorage(storage_dir="memory/test_db", backend="memory")
    
    # Add some test knowledge to the vector storage
    test_texts = [
        "NEXUS is an AI orchestration system that adapts to available resources.",
        "Machine learning models can be evaluated based on their accuracy and speed.",
        "Adaptive systems learn from past performance to improve future decisions.",
        "Large language models can process and generate human-like text.",
    ]
    vector_storage.add_texts(test_texts)
    
    # Test scenarios with different task types
    test_queries = [
        {"query": "How does NEXUS adapt to available resources?", "task_type": "general"},
        {"query": "Write code to implement a sorting algorithm", "task_type": "technical"},
        {"query": "Create a poem about artificial intelligence", "task_type": "creative"},
        {"query": "What is the capital of France?", "task_type": "factual"},
    ]
    
    # Demonstrate learning with model A
    logger.info("*** PHASE 1: Learning with first model ***")
    rag_pipeline = RAGPipeline(vector_storage=vector_storage, llm_service=model_a)
    
    logger.info("Running initial queries with model A")
    for i in range(5):
        for query_data in test_queries:
            result = await rag_pipeline.query(
                query=query_data["query"],
                task_type=query_data["task_type"]
            )
            logger.info(f"Query: {query_data['query']}")
            logger.info(f"Model used: {result['model_used']}")
            logger.info(f"Success: {'error' not in result}")
            logger.info(f"Time: {result['elapsed_time']:.2f}s")
            logger.info("---")
    
    # Show learned performance metrics
    logger.info("Performance metrics after initial learning phase:")
    performance = rag_pipeline.get_model_performance()
    for model, metrics in performance.items():
        logger.info(f"Model: {model}")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.2f}")
            else:
                logger.info(f"  {k}: {v}")
    
    # Now demonstrate adaptation by switching to model B
    logger.info("\n*** PHASE 2: Adapting to a new model ***")
    rag_pipeline.llm_service = model_b
    
    # Run more queries
    logger.info("Running queries with model B")
    for i in range(5):
        for query_data in test_queries:
            result = await rag_pipeline.query(
                query=query_data["query"],
                task_type=query_data["task_type"]
            )
            logger.info(f"Query: {query_data['query']}")
            logger.info(f"Model used: {result['model_used']}")
            logger.info(f"Success: {'error' not in result}")
            logger.info(f"Time: {result['elapsed_time']:.2f}s")
            logger.info("---")
    
    # Show updated performance metrics
    logger.info("Performance metrics after adaptation phase:")
    performance = rag_pipeline.get_model_performance()
    for model, metrics in performance.items():
        logger.info(f"Model: {model}")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.2f}")
            else:
                logger.info(f"  {k}: {v}")
    
    # Phase 3: Mixed model environment - see which one it chooses
    logger.info("\n*** PHASE 3: Smart selection between available models ***")
    
    # Create a new pipeline with both models available
    # This mock adapter will alternate between the models based on performance
    class AdaptiveService:
        def __init__(self, services):
            self.services = services
            self.current_index = 0
            self.available = True
        
        async def generate_text(self, prompt, max_tokens=1000):
            service = self.services[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.services)
            return await service.generate_text(prompt, max_tokens)
        
        async def list_models(self):
            all_models = []
            for service in self.services:
                models = await service.list_models()
                all_models.extend(models)
            return all_models
        
        async def get_available_model(self):
            service = self.services[self.current_index]
            return await service.get_available_model()
        
        async def is_model_available(self, model_name):
            for service in self.services:
                if await service.is_model_available(model_name):
                    return True
            return False
    
    # Create a new pipeline with both models
    adaptive_service = AdaptiveService([model_a, model_b])
    
    # Transfer the learned metrics
    rag_pipeline.llm_service = adaptive_service
    
    logger.info("Running final queries with adaptive selection")
    task_types = ["technical", "creative", "factual", "general"]
    
    for task_type in task_types:
        logger.info(f"\nTesting task type: {task_type}")
        for i in range(3):
            result = await rag_pipeline.query(
                query=f"This is a {task_type} query #{i+1}",
                task_type=task_type
            )
            logger.info(f"Query #{i+1}")
            logger.info(f"Model used: {result['model_used']}")
            logger.info(f"Success: {'error' not in result}")
            logger.info(f"Time: {result['elapsed_time']:.2f}s")
    
    # Final performance report
    logger.info("\nFinal performance metrics:")
    performance = rag_pipeline.get_model_performance()
    for model, metrics in performance.items():
        logger.info(f"Model: {model}")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.2f}")
            else:
                logger.info(f"  {k}: {v}")
    
    logger.info("Adaptive model selection test completed")

if __name__ == "__main__":
    asyncio.run(test_adaptive_selection())
