"""
RAG Pipeline for NEXUS
Integrates vector retrieval with generative models for knowledge-enhanced responses
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio
import json
import time
from pathlib import Path

from .vector_storage import VectorStorage
from .knowledge_manager import KnowledgeManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline that enhances LLM responses
    with relevant knowledge retrieved from vector storage.
    
    This is designed to be fully compatible with the NEXUS architecture,
    allowing dynamic selection of different LLMs, embedding models, and retrieval strategies.
    """
    
    def __init__(self, 
                vector_storage: VectorStorage = None,
                llm_service: Any = None,
                embedding_service: Any = None,
                knowledge_manager: KnowledgeManager = None):
        """Initialize the RAG pipeline
        
        Args:
            vector_storage: VectorStorage instance for knowledge retrieval
            llm_service: LLM service for generation (Gemini, Groq, etc.)
            embedding_service: Service for creating embeddings if different from vector_storage
            knowledge_manager: Optional knowledge manager to handle document processing
        """
        self.vector_storage = vector_storage or VectorStorage()
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.knowledge_manager = knowledge_manager
        
        # Performance tracking for adaptive model selection
        self.performance_metrics = {
            "response_times": {},
            "success_rates": {},
            "last_used": {}
        }
        
        # Mapping of task types to appropriate models/strategies
        self.task_strategies = {
            "general": {
                "retrieval_k": 4,
                "preferred_models": ["groq:mixtral-8x7b-32768", "gemini:gemini-1.5-pro", "ollama:llama2"]
            },
            "technical": {
                "retrieval_k": 6, 
                "preferred_models": ["groq:llama2-70b-4096", "huggingface:mistralai/Mistral-7B-Instruct-v0.2"]
            },
            "creative": {
                "retrieval_k": 3,
                "preferred_models": ["gemini:gemini-1.5-pro", "groq:claude-3-opus-20240229"]
            },
            "factual": {
                "retrieval_k": 8,
                "preferred_models": ["groq:llama2-70b-4096", "gemini:gemini-1.5-pro"]
            }
        }
    
    async def query(self, 
                  query: str,
                  task_type: str = "general",
                  k: int = None,
                  filter_dict: Dict[str, Any] = None,
                  system_prompt: str = None,
                  model_preference: str = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: The user query
            task_type: Type of task (general, technical, creative, factual)
            k: Number of knowledge chunks to retrieve (overrides task_type default)
            filter_dict: Optional filter for knowledge retrieval
            system_prompt: Optional system prompt to include
            model_preference: Optional preferred model to use
            
        Returns:
            Dict containing the generated response and related metadata
        """
        start_time = time.time()
        
        # Get strategy for this task type
        strategy = self.task_strategies.get(task_type, self.task_strategies["general"])
        
        # Use k from parameters or from strategy
        retrieval_k = k or strategy.get("retrieval_k", 4)
        
        # 1. Retrieve relevant knowledge
        knowledge_results = self.vector_storage.similarity_search(
            query, 
            k=retrieval_k, 
            filter_dict=filter_dict
        )
        
        # 2. Format knowledge for the LLM
        knowledge_context = self._format_knowledge_for_llm(knowledge_results)
        
        # 3. Select the best model for the task
        selected_model = await self._select_best_model(
            task_type, 
            model_preference, 
            strategy.get("preferred_models", [])
        )
        
        # 4. Build the complete prompt with query and knowledge
        prompt = self._build_rag_prompt(query, knowledge_context, system_prompt, task_type)
        
        # 5. Generate response with the selected LLM
        if selected_model and self.llm_service:
            response = await self._generate_with_selected_model(prompt, selected_model)
        else:
            response = {"text": "No suitable LLM service found for generation", "error": True}
        
        # 6. Record metrics for this request
        elapsed_time = time.time() - start_time
        model_name = selected_model if selected_model else "unknown"
        self._update_performance_metrics(model_name, elapsed_time, "error" not in response)
        
        # 7. Return complete results
        return {
            "query": query,
            "response": response.get("text", ""),
            "knowledge_used": knowledge_results,
            "model_used": model_name,
            "elapsed_time": elapsed_time,
            "task_type": task_type
        }
        
    def _format_knowledge_for_llm(self, knowledge_results: List[Dict[str, Any]]) -> str:
        """Format retrieved knowledge chunks for inclusion in the LLM prompt"""
        if not knowledge_results:
            return "No relevant knowledge found in the database."
            
        formatted = "Here is relevant information that may help answer the query:\n\n"
        
        for i, item in enumerate(knowledge_results, 1):
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown")
            created_at = metadata.get("created_at", "Unknown date")
            
            formatted += f"[{i}] {text}\n"
            formatted += f"Source: {source}\n"
            formatted += f"Date: {created_at}\n\n"
            
        return formatted
        
    def _build_rag_prompt(self, 
                         query: str, 
                         knowledge_context: str,
                         system_prompt: str = None,
                         task_type: str = "general") -> str:
        """Build the complete RAG prompt with query and knowledge context"""
        # Define default system prompts based on task type
        default_system_prompts = {
            "general": """You are NEXUS, an advanced AI assistant with access to retrieved knowledge.
Please answer the user's question based on the provided knowledge.
If the knowledge doesn't contain relevant information, respond based on your general knowledge,
but make it clear when you are doing so.""",

            "technical": """You are NEXUS, a technical AI assistant specialized in providing accurate
and detailed technical information. Use the retrieved knowledge to provide precise, technical answers.
Include code examples when appropriate. If the retrieved knowledge doesn't contain all necessary information,
supplement with general technical best practices.""",

            "creative": """You are NEXUS, a creative AI assistant. Use the retrieved knowledge as inspiration
to provide innovative, original responses. Think outside the box while still grounding your response
in the relevant retrieved information. Feel free to expand on ideas creatively.""",

            "factual": """You are NEXUS, a factual AI assistant focused on accuracy. Provide a detailed answer based
strictly on the retrieved knowledge. Clearly cite your sources. If the retrieved knowledge doesn't contain
sufficient information to answer completely, clearly indicate what information is missing rather than speculating."""
        }
        
        # Use provided system prompt or select based on task type
        system = system_prompt or default_system_prompts.get(task_type, default_system_prompts["general"])
        
        prompt = f"{system}\n\n"
        prompt += f"RETRIEVED KNOWLEDGE:\n{knowledge_context}\n\n"
        prompt += f"USER QUERY: {query}\n\n"
        prompt += "RESPONSE:"
        
        return prompt
    
    async def _select_best_model(self, 
                               task_type: str, 
                               model_preference: str = None,
                               preferred_models: List[str] = None) -> str:
        """
        Select the best model for the task based on availability, preference, and performance
        
        This implements the adaptive model selection aspect of the NEXUS architecture,
        learning from past performance rather than following rigid rules
        """
        # If user specified a preference and it's available, use it
        if model_preference and await self._is_model_available(model_preference):
            return model_preference
        
        # Track performance metrics over time to learn which models work best
        if not hasattr(self, 'model_performance'):
            self.model_performance = {}
        
        # Get list of models to try (preferred for this task type)
        models_to_try = preferred_models or []
        
        # If no task-specific models, use general defaults with priorities for local models first
        if not models_to_try:
            if hasattr(self.llm_service, 'available') and getattr(self.llm_service, 'available'):
                # Check the type of service we have and prioritize appropriately
                if 'ollama' in str(self.llm_service.__class__).lower():
                    # Prioritize Ollama models based on task type
                    if task_type == "technical":
                        models_to_try = ["ollama:deepseek-coder", "ollama:deepseek-r1", "groq:mixtral-8x7b-32768"]
                    elif task_type == "creative":
                        models_to_try = ["ollama:dolphin-phi", "ollama:deepseek-r1", "huggingface:mistralai/Mistral-7B-Instruct-v0.2"]
                    else:  # general or factual
                        models_to_try = ["ollama:deepseek-r1", "ollama:llama2", "groq:mixtral-8x7b-32768"]
                elif 'groq' in str(self.llm_service.__class__).lower():
                    models_to_try = ["groq:mixtral-8x7b-32768", "groq:llama2-70b-4096"]
                elif 'huggingface' in str(self.llm_service.__class__).lower():
                    models_to_try = ["huggingface:mistralai/Mistral-7B-Instruct-v0.2", "huggingface:meta-llama/Llama-2-7b-chat-hf"]
                else:
                    # Generic fallback order: Ollama first (to save API calls), then other services
                    models_to_try = ["ollama:deepseek-r1", "groq:mixtral-8x7b-32768", "gemini:gemini-1.5-pro"]
            else:
                # Default fallback if we can't detect service type
                models_to_try = ["groq:mixtral-8x7b-32768", "gemini:gemini-1.5-pro", "ollama:llama2"]
        
        # Sort models based on past performance if we have data
        if self.model_performance:
            # Create a copy of models_to_try to sort
            models_with_scores = []
            for model in models_to_try:
                # Calculate score based on success rate and speed
                perf = self.model_performance.get(model, {})
                success_rate = perf.get('success_rate', 0.5)  # Default 50% success
                avg_time = perf.get('avg_time', 1.0)  # Default 1 second
                
                # Score formula: success rate is more important than speed
                # Higher score is better
                score = (success_rate * 0.8) + (min(1.0, 1.0 / max(0.1, avg_time)) * 0.2)
                models_with_scores.append((model, score))
            
            # Sort by score, highest first
            models_with_scores.sort(key=lambda x: x[1], reverse=True)
            models_to_try = [m[0] for m in models_with_scores]
        
        # Try each model in order of preference, returning the first available one
        for model in models_to_try:
            if await self._is_model_available(model):
                return model
        
        # If we couldn't find a preferred model, attempt to get any available model
        # from the service directly
        try:
            if hasattr(self.llm_service, "get_available_model"):
                return await self.llm_service.get_available_model()
                
            # Last resort: if we have Ollama, try direct access to models
            if hasattr(self.llm_service, "list_models") and "ollama" in str(self.llm_service.__class__).lower():
                models = await self.llm_service.list_models()
                if models and len(models) > 0:
                    # Get the first available model and format it
                    model_id = models[0]["id"]
                    return f"ollama:{model_id.split(':')[0]}"  # Remove tags like :latest
        except Exception as e:
            logger.error(f"Error trying to get any available model: {e}")
        
        # No model available
        return None

    async def _is_model_available(self, model_identifier: str) -> bool:
        """
        Check if a specified model is available
        
        Args:
            model_identifier: Model identifier (format: "service:model_name")
        
        Returns:
            True if model is available, False otherwise
        """
        # Parse model identifier
        parts = model_identifier.split(":", 1)
        if len(parts) != 2:
            return False
        
        service, model_name = parts
        
        # Check availability based on service type
        try:
            if service == "groq" and hasattr(self.llm_service, "client"):
                # For Groq, check the list of available models
                models = await self.llm_service.list_available_models()
                return any(m["id"] == model_name for m in models)
                
            elif service == "huggingface" and hasattr(self.llm_service, "is_model_available"):
                # For Hugging Face, use the is_model_available method
                return await self.llm_service.is_model_available(model_name)
                
            elif service == "gemini" and hasattr(self.llm_service, "list_models"):
                # For Gemini, check if the model is in the list of available models
                # Handle both async and sync versions of list_models
                if asyncio.iscoroutinefunction(self.llm_service.list_models):
                    models = await self.llm_service.list_models()
                else:
                    models = self.llm_service.list_models()
                return model_name in models
                
            elif service == "ollama" and hasattr(self.llm_service, "list_models"):
                # For Ollama, check if the model is in the list of local models
                models = await self.llm_service.list_models()
                # Most Ollama models have a tag (like :latest) we should handle
                model_ids = [m["id"] for m in models]
                # Check if model name matches exactly or with a tag
                return model_name in model_ids or any(m.startswith(f"{model_name}:") for m in model_ids)
                
        except Exception as e:
            logger.error(f"Error checking model availability for {model_identifier}: {e}")
            
        return False
    
    async def _generate_with_selected_model(self, prompt: str, model_identifier: str) -> Dict[str, Any]:
        """
        Generate a response using the selected model
        
        Args:
            prompt: The formatted prompt
            model_identifier: The identifier of the model to use
        
        Returns:
            The generated response
        """
        # Parse model identifier
        parts = model_identifier.split(":", 1)
        if len(parts) != 2:
            return {"text": f"Invalid model identifier: {model_identifier}", "error": True}
            
        service, model_name = parts
        
        # Check if we have a service for this model type
        if not self.llm_service:
            return {"text": "No LLM service configured", "error": True}
            
        try:
            # Generate based on the service type
            if service == "groq" and hasattr(self.llm_service, "generate_text"):
                return await self.llm_service.generate_text(prompt=prompt, model=model_name)
                
            elif service == "huggingface" and hasattr(self.llm_service, "generate_text"):
                return await self.llm_service.generate_text(prompt=prompt, model_id=model_name)
                
            elif service == "gemini" and hasattr(self.llm_service, "generate_content"):
                response = await self.llm_service.generate_content(prompt, model_name)
                return {"text": response, "model": model_name}
                
            elif service == "ollama" and hasattr(self.llm_service, "generate"):
                response = await self.llm_service.generate(prompt, model=model_name)
                return {"text": response, "model": model_name}
                
            else:
                return {
                    "text": f"Service {service} doesn't support the required generation method",
                    "error": True
                }
                
        except Exception as e:
            logger.error(f"Error generating with {model_identifier}: {e}")
            return {"text": f"Error generating response: {str(e)}", "error": True}
    
    def _update_performance_metrics(self, model_name: str, elapsed_time: float, success: bool):
        """
        Update performance metrics for adaptive model selection
        
        This enables the system to learn which models perform best over time
        """
        # Initialize performance tracking if needed
        if not hasattr(self, 'model_performance'):
            self.model_performance = {}
        
        # Initialize model entry if needed
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'success_count': 0,
                'fail_count': 0,
                'total_time': 0,
                'calls': 0,
                'avg_time': 0,
                'success_rate': 0
            }
        
        # Update metrics
        perf = self.model_performance[model_name]
        perf['calls'] += 1
        perf['total_time'] += elapsed_time
        
        if success:
            perf['success_count'] += 1
        else:
            perf['fail_count'] += 1
        
        # Recalculate averages
        perf['avg_time'] = perf['total_time'] / perf['calls']
        perf['success_rate'] = perf['success_count'] / perf['calls']
        
        # Log performance update
        logger.info(f"Updated performance metrics for {model_name}: " +
                   f"Success rate: {perf['success_rate']:.2f}, Avg time: {perf['avg_time']:.2f}s")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        result = {}
        
        for model_name in self.model_performance:
            perf = self.model_performance[model_name]
            result[model_name] = {
                'success_rate': perf['success_rate'],
                'avg_time': perf['avg_time'],
                'calls': perf['calls']
            }
            
        return result
    
    async def add_document(self, 
                         document: Union[str, Dict],
                         document_id: str = None,
                         metadata: Dict[str, Any] = None,
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200) -> List[str]:
        """
        Add a document to the knowledge base for future retrieval
        
        Args:
            document: Document text or dict with text and metadata
            document_id: Optional ID for the document
            metadata: Optional metadata for the document
            chunk_size: Size of chunks to split the document into
            chunk_overlap: Amount of overlap between chunks
            
        Returns:
            List of IDs for the added chunks
        """
        # Handle document formats
        if isinstance(document, dict):
            text = document.get("text", "")
            doc_metadata = document.get("metadata", {})
            if metadata:
                doc_metadata.update(metadata)
        else:
            text = document
            doc_metadata = metadata or {}
            
        # Add source and timestamp to metadata if not present
        if "source" not in doc_metadata:
            doc_metadata["source"] = document_id or "manual_upload"
            
        if "created_at" not in doc_metadata:
            doc_metadata["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Use KnowledgeManager if available, otherwise do simple chunking
        if self.knowledge_manager:
            chunks = await self.knowledge_manager.process_document(
                text, 
                doc_metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Add chunks to vector storage
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadatas = [chunk["metadata"] for chunk in chunks]
            
            return self.vector_storage.add_texts(chunk_texts, chunk_metadatas)
        else:
            # Simple chunking if no knowledge manager
            chunks = self._simple_text_chunking(text, chunk_size, chunk_overlap)
            
            # Create metadata for each chunk (duplicate doc metadata)
            chunk_metadatas = [doc_metadata.copy() for _ in chunks]
            
            # Add chunks to vector storage
            return self.vector_storage.add_texts(chunks, chunk_metadatas)
    
    def _simple_text_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Simple text chunking strategy"""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position for next chunk, considering overlap
            start = start + chunk_size - chunk_overlap
            
        return chunks
