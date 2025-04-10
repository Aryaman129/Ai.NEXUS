"""
NEXUS Standalone RAG Demo
Demonstrates the RAG pipeline with Hugging Face and Groq integrations
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

class SimpleLLMManager:
    """A simple manager for LLM services"""
    
    def __init__(self):
        """Initialize the LLM manager"""
        self.services = {}
        
    def register_service(self, name, service):
        """Register an LLM service"""
        self.services[name] = service
        
    async def generate_with_best_service(self, prompt, model_preference=None):
        """Generate text with the best available service"""
        # Try services in order of preference
        service_order = ["groq", "huggingface", "gemini"]
        
        # If a model preference is specified, try to find a service that can use it
        if model_preference:
            for service_name, service in self.services.items():
                # Different services have different ways of checking model availability
                try:
                    if hasattr(service, "is_model_available"):
                        model_available = await service.is_model_available(model_preference)
                        if model_available:
                            logger.info(f"Using {service_name} with model {model_preference}")
                            if service_name == "groq":
                                return await service.generate_text(prompt=prompt, model=model_preference)
                            elif service_name == "huggingface":
                                return await service.generate_text(prompt=prompt, model_id=model_preference)
                            else:
                                return await service.generate_text(prompt=prompt)
                except Exception as e:
                    logger.warning(f"Error checking model availability with {service_name}: {e}")
        
        # Try services in order of preference
        for service_name in service_order:
            if service_name in self.services:
                try:
                    logger.info(f"Trying {service_name} service")
                    service = self.services[service_name]
                    
                    # Different services have different method signatures
                    if service_name == "groq":
                        return await service.generate_text(prompt=prompt)
                    elif service_name == "huggingface":
                        return await service.generate_text(prompt=prompt)
                    elif service_name == "gemini":
                        result = await service.generate_text(prompt=prompt)
                        return {"text": result.get("text", "")}
                except Exception as e:
                    logger.warning(f"Error with {service_name} service: {e}")
        
        return {"text": "No LLM service available", "error": True}

class SimpleRAGPipeline:
    """A simple RAG pipeline for demonstration purposes"""
    
    def __init__(self, vector_store, llm_manager):
        """Initialize the simple RAG pipeline"""
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        
    async def query(self, query, k=4, model_preference=None):
        """Process a query through the RAG pipeline"""
        # Track timing
        start_time = time.time()
        
        # 1. Retrieve relevant documents
        results = self.vector_store.similarity_search(query, k=k)
        
        # 2. Format the retrieved documents for the LLM
        knowledge_context = self._format_knowledge_for_llm(results)
        
        # 3. Build the prompt with the query and knowledge
        prompt = self._build_rag_prompt(query, knowledge_context)
        
        # 4. Generate a response with the LLM
        llm_response = await self.llm_manager.generate_with_best_service(prompt, model_preference)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # 5. Return the results
        return {
            "query": query,
            "response": llm_response.get("text", "No response generated"),
            "knowledge_used": results,
            "elapsed_time": elapsed_time,
            "model_used": llm_response.get("model", "Unknown")
        }
    
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
    
    def _build_rag_prompt(self, query, knowledge_context):
        """Build the complete RAG prompt"""
        prompt = """You are NEXUS, an advanced AI assistant with access to retrieved knowledge.
Please answer the user's question based on the provided knowledge.
If the knowledge doesn't contain relevant information, respond based on your general knowledge,
but make it clear when you are doing so.

RETRIEVED KNOWLEDGE:
"""
        
        prompt += knowledge_context
        prompt += f"\n\nUSER QUERY: {query}\n\n"
        prompt += "RESPONSE:"
        
        return prompt

async def main():
    """Main entry point for the RAG demo"""
    print("\n" + "="*80)
    print("NEXUS Standalone RAG Demo")
    print("="*80 + "\n")
    
    # Initialize the LLM services
    print("Initializing LLM services...")
    
    llm_manager = SimpleLLMManager()
    
    # Initialize and register Groq
    groq_service = GroqIntegration()
    llm_manager.register_service("groq", groq_service)
    print(f"Registered Groq service (available: {groq_service.available})")
    
    # Initialize and register Hugging Face
    hf_service = HuggingFaceIntegration()
    llm_manager.register_service("huggingface", hf_service)
    print(f"Registered Hugging Face service (available: {hf_service.available})")
    
    # Initialize the vector store and add sample data
    print("\nInitializing vector store with sample knowledge...")
    vector_store = SimpleVectorStore()
    
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
    
    vector_store.add_texts(sample_texts, sample_metadata)
    print(f"Added {len(sample_texts)} knowledge items to vector store")
    
    # Initialize the RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag_pipeline = SimpleRAGPipeline(vector_store, llm_manager)
    
    # Demo queries
    print("\nProcessing demo queries with RAG pipeline...\n")
    
    demo_queries = [
        "What is NEXUS?",
        "Explain how RAG works",
        "Tell me about language models"
    ]
    
    for query in demo_queries:
        print(f"\n>>> Query: {query}")
        print("-" * 50)
        
        result = await rag_pipeline.query(query)
        
        print(f"Response:")
        print(result["response"])
        print(f"\nUsed model: {result.get('model_used', 'Unknown')}")
        print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
        print("-" * 50)
    
    print("\nDemo completed!\n")

if __name__ == "__main__":
    asyncio.run(main())
