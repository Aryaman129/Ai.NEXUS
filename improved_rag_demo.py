"""
NEXUS Improved RAG Demo
Demonstrates the RAG pipeline with Hugging Face, Groq, and Ollama integrations
"""
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import json
import time
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the integrations we've created
from src.integrations.groq_integration import GroqIntegration
from src.integrations.huggingface_integration import HuggingFaceIntegration
from src.integrations.ollama_integration import OllamaIntegration

class SimpleVectorStore:
    """A simplified vector storage for demonstration purposes"""
    
    def __init__(self):
        """Initialize the simple vector store"""
        self.texts = []
        self.metadatas = []
        self.embeddings = []
        
    def add_texts(self, texts, metadatas=None):
        """Add texts to the vector store"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Create simple embeddings (random for demo purposes)
        # In a real system, we'd use a proper embedding model
        embeddings = [self._simple_embedding(text) for text in texts]
        
        # Store the texts, metadatas, and embeddings
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embeddings)
        
        return [f"id_{i}" for i in range(len(self.texts) - len(texts), len(self.texts))]
        
    def similarity_search(self, query, k=4, filter_dict=None):
        """Search for similar texts"""
        if not self.texts:
            return []
            
        # Create a simple embedding for the query
        query_embedding = self._simple_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            # Apply filter if provided
            if filter_dict and not self._matches_filter(self.metadatas[i], filter_dict):
                continue
                
            similarities.append((i, similarity))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top k results
        results = []
        for i, similarity in similarities[:k]:
            results.append({
                "text": self.texts[i],
                "metadata": self.metadatas[i],
                "id": f"id_{i}",
                "score": similarity
            })
            
        return results
    
    def _simple_embedding(self, text):
        """Create a simple embedding for a text (for demo purposes only)"""
        # In a real system, we'd use a proper embedding model
        # This is just for demonstration
        import hashlib
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert the hash to a fixed-size vector
        vector = []
        for i in range(0, len(text_hash), 2):
            if i < len(text_hash) - 1:
                value = int(text_hash[i:i+2], 16)
                vector.append(value / 255.0)  # Normalize to [0, 1]
                
        # Pad or truncate to a fixed size
        vector_size = 20
        if len(vector) < vector_size:
            vector.extend([0.0] * (vector_size - len(vector)))
        else:
            vector = vector[:vector_size]
            
        return vector
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _matches_filter(self, metadata, filter_dict):
        """Check if metadata matches the filter"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

class AdaptiveLLMManager:
    """A dynamic manager for LLM services that adapts to what's available"""
    
    def __init__(self):
        """Initialize the LLM manager"""
        self.services = {}
        self.service_stats = {}
        
    def register_service(self, name, service):
        """Register an LLM service"""
        self.services[name] = service
        self.service_stats[name] = {
            "success_count": 0,
            "failure_count": 0,
            "total_time": 0,
            "call_count": 0
        }
        
    async def list_all_models(self):
        """List all available models across all services"""
        all_models = {}
        
        for service_name, service in self.services.items():
            try:
                if hasattr(service, "list_available_models") and callable(service.list_available_models):
                    models = await service.list_available_models()
                    all_models[service_name] = models
                elif hasattr(service, "list_models") and callable(service.list_models):
                    models = await service.list_models()
                    all_models[service_name] = models
            except Exception as e:
                logger.warning(f"Error listing models for {service_name}: {e}")
                
        return all_models
        
    async def generate_with_best_service(self, prompt, task_type=None, model_preference=None):
        """Generate text with the best available service based on task and preferences"""
        start_time = time.time()
        
        # Get adaptive service order based on task type
        service_order = self._get_service_order_for_task(task_type)
        
        # If a model preference is specified, try to find a service that can use it
        if model_preference:
            service_name, model_name = self._parse_model_preference(model_preference)
            
            if service_name and service_name in self.services:
                # User specified both service and model
                try:
                    result = await self._generate_with_service(service_name, prompt, model_name)
                    self._update_stats(service_name, result, time.time() - start_time)
                    return result
                except Exception as e:
                    logger.warning(f"Error with preferred service {service_name}: {e}")
            elif model_name:
                # User only specified model name, try to find a service that supports it
                for service_name in service_order:
                    if service_name in self.services:
                        try:
                            result = await self._generate_with_service(service_name, prompt, model_name)
                            self._update_stats(service_name, result, time.time() - start_time)
                            return result
                        except Exception:
                            pass
        
        # Try services in adaptive order
        for service_name in service_order:
            if service_name in self.services:
                try:
                    logger.info(f"Trying {service_name} service")
                    result = await self._generate_with_service(service_name, prompt)
                    self._update_stats(service_name, result, time.time() - start_time)
                    return result
                except Exception as e:
                    logger.warning(f"Error with {service_name} service: {e}")
        
        return {"text": "No LLM service available", "error": True}
    
    def _parse_model_preference(self, preference):
        """Parse a model preference into service and model name"""
        if ":" in preference:
            parts = preference.split(":", 1)
            return parts[0], parts[1]
        else:
            return None, preference
    
    def _get_service_order_for_task(self, task_type):
        """Get the optimal service order for a given task type"""
        # Define task-specific service orders
        task_service_orders = {
            "general": ["groq", "ollama", "huggingface"],
            "creative": ["groq", "huggingface", "ollama"],
            "factual": ["groq", "ollama", "huggingface"],
            "technical": ["ollama", "groq", "huggingface"],
            "code": ["ollama", "groq", "huggingface"]
        }
        
        # Use success rates to influence ordering if we have enough data
        if all(stats["call_count"] > 5 for stats in self.service_stats.values()):
            # Calculate success rates
            success_rates = {}
            for name, stats in self.service_stats.items():
                if stats["call_count"] > 0:
                    success_rates[name] = stats["success_count"] / stats["call_count"]
                else:
                    success_rates[name] = 0
                    
            # Sort services by success rate (descending)
            sorted_services = sorted(success_rates.items(), key=lambda x: x[1], reverse=True)
            return [name for name, _ in sorted_services]
        
        # Otherwise, use predefined order based on task type
        return task_service_orders.get(task_type, ["groq", "ollama", "huggingface"])
    
    async def _generate_with_service(self, service_name, prompt, model=None):
        """Generate text with a specific service"""
        service = self.services[service_name]
        
        # Different services have different method signatures and model parameter names
        if service_name == "groq":
            kwargs = {"prompt": prompt}
            if model:
                kwargs["model"] = model
            return await service.generate_text(**kwargs)
            
        elif service_name == "huggingface":
            kwargs = {"prompt": prompt}
            if model:
                kwargs["model_id"] = model
            return await service.generate_text(**kwargs)
            
        elif service_name == "ollama":
            kwargs = {"prompt": prompt}
            if model:
                kwargs["model"] = model
            return await service.generate_text(**kwargs)
            
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def _update_stats(self, service_name, result, elapsed_time):
        """Update statistics for a service"""
        stats = self.service_stats[service_name]
        stats["call_count"] += 1
        stats["total_time"] += elapsed_time
        
        if result and not result.get("error", False):
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
    
    def get_stats(self):
        """Get statistics for all services"""
        stats = {}
        for name, service_stats in self.service_stats.items():
            if service_stats["call_count"] > 0:
                stats[name] = {
                    "success_rate": service_stats["success_count"] / service_stats["call_count"],
                    "avg_response_time": service_stats["total_time"] / service_stats["call_count"],
                    "call_count": service_stats["call_count"]
                }
            else:
                stats[name] = {
                    "success_rate": 0,
                    "avg_response_time": 0,
                    "call_count": 0
                }
        return stats

class AdaptiveRAGPipeline:
    """An adaptive RAG pipeline that dynamically selects strategies based on the query"""
    
    def __init__(self, vector_store, llm_manager):
        """Initialize the adaptive RAG pipeline"""
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        
        # Strategy configuration
        self.strategies = {
            "default": {
                "k": 4,
                "model_preference": None,
                "prompt_template": self._default_prompt_template
            },
            "technical": {
                "k": 6,
                "model_preference": "groq:llama3-70b-8192",
                "prompt_template": self._technical_prompt_template
            },
            "creative": {
                "k": 3,
                "model_preference": "groq:meta-llama/llama-4-scout-17b-16e-instruct",
                "prompt_template": self._creative_prompt_template
            },
            "factual": {
                "k": 6,
                "model_preference": "groq:gemma2-9b-it",
                "prompt_template": self._factual_prompt_template
            }
        }
        
    async def query(self, query, strategy="default", k=None, model_preference=None):
        """Process a query through the RAG pipeline with a specific strategy"""
        # Track timing
        start_time = time.time()
        
        # Auto-select strategy based on query content if not specified
        if strategy == "auto":
            strategy = self._detect_query_strategy(query)
        
        # Get strategy configuration
        strategy_config = self.strategies.get(strategy, self.strategies["default"])
        
        # Override strategy parameters if provided
        retrieval_k = k if k is not None else strategy_config["k"]
        final_model_preference = model_preference or strategy_config["model_preference"]
        
        # 1. Retrieve relevant documents
        results = self.vector_store.similarity_search(query, k=retrieval_k)
        
        # 2. Format the retrieved documents for the LLM
        knowledge_context = self._format_knowledge_for_llm(results)
        
        # 3. Build the prompt with the query and knowledge using the strategy's template
        prompt = strategy_config["prompt_template"](query, knowledge_context)
        
        # 4. Generate a response with the LLM
        llm_response = await self.llm_manager.generate_with_best_service(
            prompt, 
            task_type=strategy,
            model_preference=final_model_preference
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # 5. Return the results
        return {
            "query": query,
            "response": llm_response.get("text", "No response generated"),
            "knowledge_used": results,
            "elapsed_time": elapsed_time,
            "model_used": llm_response.get("model", "Unknown"),
            "strategy": strategy
        }
    
    def _detect_query_strategy(self, query):
        """Automatically detect the best strategy for a query"""
        query_lower = query.lower()
        
        # Check for technical indicators
        technical_keywords = ["how to", "explain", "what is", "define", "code", "implementation"]
        if any(keyword in query_lower for keyword in technical_keywords):
            return "technical"
            
        # Check for creative indicators
        creative_keywords = ["generate", "create", "design", "imagine", "story", "ideas", "creative"]
        if any(keyword in query_lower for keyword in creative_keywords):
            return "creative"
            
        # Check for factual indicators
        factual_keywords = ["when", "where", "who", "factual", "facts", "history", "date", "information"]
        if any(keyword in query_lower for keyword in factual_keywords):
            return "factual"
            
        # Default strategy
        return "default"
    
    def _format_knowledge_for_llm(self, knowledge_results):
        """Format retrieved knowledge for the LLM"""
        if not knowledge_results:
            return "No relevant knowledge found."
            
        formatted = "Here is relevant information that may help answer the query:\n\n"
        
        for i, item in enumerate(knowledge_results, 1):
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            formatted += f"[{i}] {text}\n"
            formatted += f"Source: {source}\n\n"
            
        return formatted
    
    def _default_prompt_template(self, query, knowledge_context):
        """Default prompt template for general queries"""
        return f"""You are NEXUS, an advanced AI assistant with access to retrieved knowledge.
Please answer the user's question based on the provided knowledge.
If the knowledge doesn't contain relevant information, respond based on your general knowledge,
but make it clear when you are doing so.

RETRIEVED KNOWLEDGE:
{knowledge_context}

USER QUERY: {query}

RESPONSE:"""
    
    def _technical_prompt_template(self, query, knowledge_context):
        """Prompt template for technical queries"""
        return f"""You are NEXUS, a technical AI assistant specialized in providing accurate
and detailed technical information. Use the retrieved knowledge to provide precise, technical answers.
Include code examples when appropriate. If the retrieved knowledge doesn't contain all necessary information,
supplement with general technical best practices.

RETRIEVED KNOWLEDGE:
{knowledge_context}

USER QUERY: {query}

RESPONSE:"""
    
    def _creative_prompt_template(self, query, knowledge_context):
        """Prompt template for creative queries"""
        return f"""You are NEXUS, a creative AI assistant. Use the retrieved knowledge as inspiration
to provide innovative, original responses. Think outside the box while still grounding your response
in the relevant retrieved information. Feel free to expand on ideas creatively.

RETRIEVED KNOWLEDGE:
{knowledge_context}

USER QUERY: {query}

RESPONSE:"""
    
    def _factual_prompt_template(self, query, knowledge_context):
        """Prompt template for factual queries"""
        return f"""You are NEXUS, a factual AI assistant focused on accuracy. Provide a detailed answer based
strictly on the retrieved knowledge. Clearly cite your sources. If the retrieved knowledge doesn't contain
sufficient information to answer completely, clearly indicate what information is missing rather than speculating.

RETRIEVED KNOWLEDGE:
{knowledge_context}

USER QUERY: {query}

RESPONSE:"""

async def check_ollama_models():
    """Check what Ollama models are available locally"""
    print("Checking Ollama models...")
    ollama_service = OllamaIntegration()
    
    # Wait a bit for Ollama to initialize
    await asyncio.sleep(2)
    
    if ollama_service.available:
        models = await ollama_service.list_models()
        if models:
            print(f"Found {len(models)} Ollama models:")
            for model in models:
                print(f"  - {model['id']}")
        else:
            print("No Ollama models found.")
    else:
        print("Ollama service not available. Is Ollama running?")
    
    return ollama_service

async def main():
    """Main entry point for the RAG demo"""
    print("\n" + "="*80)
    print("NEXUS Improved RAG Demo")
    print("="*80 + "\n")
    
    # Initialize the LLM services
    print("Initializing LLM services...")
    
    llm_manager = AdaptiveLLMManager()
    
    # Initialize and register Groq
    groq_service = GroqIntegration()
    llm_manager.register_service("groq", groq_service)
    print(f"Registered Groq service (available: {groq_service.available})")
    
    # Initialize and register Hugging Face
    hf_service = HuggingFaceIntegration()
    llm_manager.register_service("huggingface", hf_service)
    print(f"Registered Hugging Face service (available: {hf_service.available})")
    
    # Initialize and register Ollama (if available)
    ollama_service = await check_ollama_models()
    if ollama_service.available:
        llm_manager.register_service("ollama", ollama_service)
        print("Registered Ollama service for local models")
    
    # List available models
    print("\nListing available models across services...")
    all_models = await llm_manager.list_all_models()
    for service_name, models in all_models.items():
        if models:
            print(f"\n{service_name.capitalize()} models ({len(models)}):")
            for i, model in enumerate(models[:5], 1):  # Show first 5 models
                print(f"  {i}. {model.get('id', 'unknown')}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
    
    # Initialize the vector store and add sample data
    print("\nInitializing vector store with sample knowledge...")
    vector_store = SimpleVectorStore()
    
    sample_texts = [
        "NEXUS is an AI orchestration system that dynamically selects and combines tools.",
        "The RAG (Retrieval-Augmented Generation) paradigm enhances LLM responses with relevant knowledge.",
        "Hugging Face provides access to thousands of AI models for various tasks.",
        "Groq is known for its fast inference speeds for large language models.",
        "Python is a popular programming language for AI and machine learning.",
        "The transformer architecture revolutionized natural language processing.",
        "Ollama allows running open-source large language models locally on your computer.",
        "Language models like Llama and Mistral can be fine-tuned for specific tasks.",
        "Vector embeddings convert text into numerical representations that capture semantic meaning."
    ]
    
    sample_metadata = [
        {"source": "NEXUS Documentation", "topic": "architecture"},
        {"source": "AI Research Paper", "topic": "rag"},
        {"source": "API Documentation", "topic": "huggingface"},
        {"source": "API Documentation", "topic": "groq"},
        {"source": "Programming Guide", "topic": "python"},
        {"source": "AI Research Paper", "topic": "transformers"},
        {"source": "Ollama Documentation", "topic": "local_models"},
        {"source": "AI Capabilities Guide", "topic": "fine_tuning"},
        {"source": "Vector Database Guide", "topic": "embeddings"}
    ]
    
    vector_store.add_texts(sample_texts, sample_metadata)
    print(f"Added {len(sample_texts)} knowledge items to vector store")
    
    # Initialize the RAG pipeline
    print("\nInitializing Adaptive RAG pipeline...")
    rag_pipeline = AdaptiveRAGPipeline(vector_store, llm_manager)
    
    # Demo queries
    print("\nProcessing demo queries with Adaptive RAG pipeline...\n")
    
    demo_queries = [
        {"query": "What is NEXUS?", "strategy": "factual"},
        {"query": "Explain how RAG works", "strategy": "technical"},
        {"query": "Tell me about language models", "strategy": "default"},
        {"query": "Generate a creative story about AI tools working together", "strategy": "creative"}
    ]
    
    for query_info in demo_queries:
        query = query_info["query"]
        strategy = query_info["strategy"]
        
        print(f"\n>>> Query: {query}")
        print(f">>> Strategy: {strategy}")
        print("-" * 50)
        
        result = await rag_pipeline.query(query, strategy=strategy)
        
        print(f"Response:")
        print(result["response"])
        print(f"\nUsed model: {result.get('model_used', 'Unknown')}")
        print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
        print("-" * 50)
    
    # Auto strategy detection demo
    print("\n>>> Demonstrating automatic strategy detection...")
    
    auto_queries = [
        "How does Python handle garbage collection?",
        "Create an imaginative scenario where AI helps solve climate change",
        "When was the transformer architecture first introduced?"
    ]
    
    for query in auto_queries:
        print(f"\n>>> Query: {query}")
        print(f">>> Strategy: auto")
        print("-" * 50)
        
        result = await rag_pipeline.query(query, strategy="auto")
        
        print(f"Detected strategy: {result['strategy']}")
        print(f"Response:")
        print(result["response"])
        print(f"\nUsed model: {result.get('model_used', 'Unknown')}")
        print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
        print("-" * 50)
    
    # Show service statistics
    print("\nService usage statistics:")
    stats = llm_manager.get_stats()
    for service_name, service_stats in stats.items():
        print(f"  {service_name}:")
        print(f"    Success rate: {service_stats['success_rate']:.2f}")
        print(f"    Avg response time: {service_stats['avg_response_time']:.2f} seconds")
        print(f"    Total calls: {service_stats['call_count']}")
    
    print("\nDemo completed!\n")

if __name__ == "__main__":
    asyncio.run(main())
