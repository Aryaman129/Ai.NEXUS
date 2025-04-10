"""
Adaptive LLM Manager

This module provides an adaptive manager for LLM providers that selects
the best provider and model based on task type and performance history.
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
import asyncio

from .llm_registry import LLMRegistry
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class AdaptiveLLMManager:
    """
    Adaptive manager for LLM providers
    
    This class selects the best provider and model based on task type
    and performance history, embodying the NEXUS philosophy of adaptation
    over rigid rules.
    """
    
    def __init__(self, registry: LLMRegistry, memory_path: str = "memory/llm"):
        self.registry = registry
        self.memory_path = memory_path
        
        # Task-specific preferences
        self.task_preferences = {
            "technical": {
                "preferred_models": [
                    "huggingface:mistralai/Mistral-7B-Instruct-v0.2",
                    "ollama:deepseek-coder",
                    "groq:mixtral-8x7b-32768"
                ],
                "temperature": 0.3,
                "max_tokens": 2048
            },
            "creative": {
                "preferred_models": [
                    "ollama:llama2",
                    "groq:llama2-70b-4096",
                    "gemini:gemini-pro"
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "factual": {
                "preferred_models": [
                    "groq:mixtral-8x7b-32768",
                    "ollama:deepseek-coder",
                    "mistral:mistral-medium"
                ],
                "temperature": 0.1,
                "max_tokens": 1024
            },
            "general": {
                "preferred_models": [
                    "ollama:llama2",
                    "groq:mixtral-8x7b-32768",
                    "gemini:gemini-pro"
                ],
                "temperature": 0.5,
                "max_tokens": 1024
            }
        }
        
        # Task performance metrics
        self.task_performance = {}
        
        # Load task performance if available
        self._load_task_performance()
        
    async def generate_text(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          task_type: str = "general",
                          model_preference: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate text using the best model for the task
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context
            task_type: Type of task (technical, creative, factual, general)
            model_preference: Optional specific model preference
            **kwargs: Additional parameters
            
        Returns:
            Generated text and metadata
        """
        # Select the best model for this task
        provider_name, model = await self._select_best_model(task_type, model_preference)
        
        if not provider_name or not model:
            logger.warning(f"No suitable model found for task type: {task_type}")
            return {"error": f"No suitable model found for task type: {task_type}"}
            
        # Get task-specific parameters
        task_params = self.task_preferences.get(task_type, {})
        
        # Combine parameters (priority: kwargs > task_params)
        combined_params = {
            "temperature": task_params.get("temperature", 0.5),
            "max_tokens": task_params.get("max_tokens", 1024)
        }
        combined_params.update(kwargs)
        
        # Start timing
        start_time = time.time()
        
        # Generate text
        result = await self.registry.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            provider_name=provider_name,
            model=model,
            **combined_params
        )
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        success = "text" in result and "error" not in result
        
        # Update task performance
        await self._update_task_performance(
            task_type=task_type,
            provider_name=provider_name,
            model=model,
            success=success,
            latency=elapsed_time,
            tokens=result.get("tokens", 0)
        )
        
        # Add task type to result
        result["task_type"] = task_type
        result["provider"] = provider_name
        
        return result
        
    async def _select_best_model(self, 
                               task_type: str, 
                               model_preference: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Select the best model for a task
        
        Args:
            task_type: Type of task
            model_preference: Optional specific model preference
            
        Returns:
            Tuple of (provider_name, model)
        """
        # Honor explicit preference if available
        if model_preference:
            # Parse provider and model
            if ":" in model_preference:
                provider_name, model = model_preference.split(":", 1)
                
                # Check if provider is available
                provider = await self.registry.get_provider(provider_name)
                if provider and await provider.is_available():
                    return provider_name, model
            else:
                # Try to find a provider for this model
                for provider_name, provider in self.registry.providers.items():
                    if await provider.is_available():
                        # Check if this provider has the model
                        models = await provider.get_available_models()
                        for model_info in models:
                            if model_info["id"] == model_preference:
                                return provider_name, model_preference
        
        # Get task-specific preferences
        task_prefs = self.task_preferences.get(task_type, {})
        preferred_models = task_prefs.get("preferred_models", [])
        
        # Try preferred models first
        for model_id in preferred_models:
            if ":" in model_id:
                provider_name, model = model_id.split(":", 1)
                
                # Check if provider is available
                provider = await self.registry.get_provider(provider_name)
                if provider and await provider.is_available():
                    return provider_name, model
        
        # If no preferred models are available, find the best performing model for this task
        if task_type in self.task_performance:
            task_perf = self.task_performance[task_type]
            
            # Sort models by score
            scored_models = []
            
            for model_id, metrics in task_perf.items():
                if ":" in model_id:
                    provider_name, model = model_id.split(":", 1)
                    
                    # Check if provider is available
                    provider = await self.registry.get_provider(provider_name)
                    if provider and await provider.is_available():
                        # Calculate score
                        score = self._calculate_model_score(metrics)
                        scored_models.append((provider_name, model, score))
            
            # Sort by score (highest first)
            scored_models.sort(key=lambda x: x[2], reverse=True)
            
            if scored_models:
                return scored_models[0][0], scored_models[0][1]
        
        # If all else fails, use any available provider
        for provider_name, provider in self.registry.providers.items():
            if await provider.is_available():
                # Get available models
                models = await provider.get_available_models()
                if models:
                    return provider_name, models[0]["id"]
        
        # No suitable model found
        return None, None
        
    def _calculate_model_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate a score for model selection based on performance metrics
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Score (higher = better)
        """
        if not metrics or metrics.get("calls", 0) == 0:
            return 0.0
            
        # Base score from success rate (0-100)
        success_rate = metrics.get("successes", 0) / metrics.get("calls", 1) * 100
        score = success_rate
        
        # Penalize for high latency
        latency_penalty = min(20, metrics.get("average_latency", 0))
        score -= latency_penalty
        
        # Bonus for more experience (calls)
        experience_bonus = min(10, metrics.get("calls", 0) / 10)
        score += experience_bonus
        
        return score
        
    async def _update_task_performance(self, 
                                     task_type: str, 
                                     provider_name: str,
                                     model: str,
                                     success: bool,
                                     latency: float,
                                     tokens: int) -> None:
        """
        Update task performance metrics
        
        Args:
            task_type: Type of task
            provider_name: Provider name
            model: Model ID
            success: Whether the generation was successful
            latency: Generation latency
            tokens: Token usage
        """
        # Initialize task performance if needed
        if task_type not in self.task_performance:
            self.task_performance[task_type] = {}
            
        # Create model ID
        model_id = f"{provider_name}:{model}"
        
        # Initialize model metrics if needed
        if model_id not in self.task_performance[task_type]:
            self.task_performance[task_type][model_id] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_tokens": 0,
                "average_latency": 0.0
            }
            
        # Update metrics
        metrics = self.task_performance[task_type][model_id]
        metrics["calls"] += 1
        
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
            
        metrics["total_tokens"] += tokens
        
        # Exponential moving average for latency
        alpha = 0.1  # Weight for new values
        metrics["average_latency"] = (alpha * latency + 
                                    (1 - alpha) * metrics["average_latency"])
                                    
        # Save task performance periodically
        if metrics["calls"] % 10 == 0:
            await self._save_task_performance()
            
    async def _save_task_performance(self) -> None:
        """Save task performance metrics to disk"""
        try:
            os.makedirs(self.memory_path, exist_ok=True)
            
            performance_path = os.path.join(self.memory_path, "task_performance.json")
            with open(performance_path, 'w') as f:
                json.dump(self.task_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving task performance: {e}")
            
    def _load_task_performance(self) -> None:
        """Load task performance metrics from disk"""
        try:
            performance_path = os.path.join(self.memory_path, "task_performance.json")
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.task_performance = json.load(f)
        except Exception as e:
            logger.error(f"Error loading task performance: {e}")
            self.task_performance = {}
