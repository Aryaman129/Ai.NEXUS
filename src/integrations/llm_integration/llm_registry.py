"""
LLM Registry

This module provides a registry for LLM providers with adaptive selection
based on performance metrics and availability.
"""

import time
import logging
import json
import os
from typing import Dict, List, Any, Optional, Type, Tuple
import asyncio

from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class LLMRegistry:
    """Registry for LLM providers with adaptive selection"""
    
    def __init__(self, memory_path: str = "memory/llm"):
        self.providers = {}  # name -> provider instance
        self.performance_metrics = {}  # name -> metrics
        self.provider_versions = {}  # name -> version history
        self.provider_classes = {}  # name -> class
        self.memory_path = memory_path
        
        # Parameter mappings for standardization
        self.parameter_mappings = {
            "ollama": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "model": "model"
            },
            "groq": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "model": "model"
            },
            "huggingface": {
                "temperature": "temperature",
                "max_tokens": "max_length",
                "top_p": "top_p",
                "model": "model_id"
            },
            "mistral": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "model": "model"
            },
            "openrouter": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "model": "model"
            },
            "together": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "model": "model"
            },
            "gemini": {
                "temperature": "temperature",
                "max_tokens": "max_output_tokens",
                "top_p": "top_p",
                "model": "model"
            }
        }
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Load performance metrics if available
        self._load_performance_metrics()
        
    def register_provider(self, name: str, provider_class: Type[LLMInterface], 
                         priority: int = 0,
                         config: Optional[Dict] = None) -> bool:
        """
        Register an LLM provider
        
        Args:
            name: Unique provider name
            provider_class: Provider class (not instance)
            priority: Priority (higher = preferred)
            config: Optional configuration
            
        Returns:
            Success status
        """
        # Store the class for later instantiation
        self.provider_classes[name] = {
            "class": provider_class,
            "priority": priority,
            "config": config or {}
        }
        
        # Initialize metrics if needed
        if name not in self.performance_metrics:
            self.performance_metrics[name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_tokens": 0,
                "average_latency": 0.0,
                "models": {}
            }
            
        # Initialize version history
        if name not in self.provider_versions:
            self.provider_versions[name] = []
            
        logger.info(f"Registered LLM provider: {name}")
        return True
        
    async def initialize_provider(self, name: str) -> Optional[LLMInterface]:
        """
        Initialize a specific provider
        
        Args:
            name: Provider name
            
        Returns:
            Initialized provider or None if failed
        """
        if name not in self.provider_classes:
            logger.warning(f"Provider {name} not registered")
            return None
            
        # Create instance
        provider_info = self.provider_classes[name]
        provider_class = provider_info["class"]
        config = provider_info["config"]
        
        try:
            # Create provider instance
            provider = provider_class(config)
            
            # Try to initialize
            success = await provider.initialize(config)
            if success:
                # Store instance
                self.providers[name] = provider
                    
                # Record version
                capabilities = provider.get_capabilities()
                version = capabilities.get("version", "unknown")
                self.provider_versions[name].append({
                    "version": version,
                    "timestamp": time.time(),
                    "capabilities": capabilities
                })
                
                logger.info(f"Initialized LLM provider {name} v{version}")
                return provider
            else:
                logger.warning(f"Failed to initialize LLM provider {name}")
                return None
        except Exception as e:
            logger.error(f"Error initializing LLM provider {name}: {e}")
            return None
        
    async def get_provider(self, name: Optional[str] = None,
                         required_capabilities: Optional[List[str]] = None) -> Optional[LLMInterface]:
        """
        Get a provider by name or capabilities
        
        Args:
            name: Optional specific provider name
            required_capabilities: Optional list of required capabilities
            
        Returns:
            Best matching provider or None if none available
        """
        # Case 1: Specific provider requested
        if name:
            # Try existing providers
            if name in self.providers:
                return self.providers[name]
                
            # Not found, try to initialize
            return await self.initialize_provider(name)
            
        # Case 2: Provider with required capabilities
        best_provider = None
        best_score = -1
        
        for provider_name, provider in self.providers.items():
            # Check capabilities if required
            if required_capabilities:
                capabilities = provider.get_capabilities()
                if not all(cap in capabilities for cap in required_capabilities):
                    continue
                    
            score = self._calculate_provider_score(provider_name)
            if score > best_score:
                best_score = score
                best_provider = provider
                
        return best_provider
        
    def _calculate_provider_score(self, provider_name: str) -> float:
        """
        Calculate a score for provider selection based on performance metrics
        
        Higher score = preferred provider
        """
        if provider_name not in self.performance_metrics:
            return 0.0
            
        metrics = self.performance_metrics[provider_name]
        
        # Base score from priority
        priority = self.provider_classes[provider_name]["priority"]
        score = priority * 10  # Priority is a major factor
        
        # Add performance metrics if we have sufficient data
        if metrics["calls"] > 0:
            # Success rate (0-100)
            success_rate = metrics["successes"] / metrics["calls"] * 100
            score += success_rate
            
            # Penalize for high latency
            latency_penalty = min(20, metrics["average_latency"])
            score -= latency_penalty
            
        return score
        
    async def update_metrics(self, provider_name: str, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for a provider"""
        if provider_name not in self.performance_metrics:
            logger.warning(f"Provider {provider_name} not in registry, can't update metrics")
            return
            
        current = self.performance_metrics[provider_name]
        
        # Update calls and success counts
        current["calls"] += 1
        if metrics.get("success", False):
            current["successes"] += 1
        else:
            current["failures"] += 1
            
        # Update token metrics
        tokens = metrics.get("tokens", 0)
        current["total_tokens"] += tokens
            
        # Update latency metrics
        if "latency" in metrics:
            # Exponential moving average for latency
            alpha = 0.1  # Weight for new values
            current["average_latency"] = (alpha * metrics["latency"] + 
                                        (1 - alpha) * current["average_latency"])
                                        
        # Update model-specific metrics
        model = metrics.get("model")
        if model:
            if model not in current["models"]:
                current["models"][model] = {
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_tokens": 0,
                    "average_latency": 0.0
                }
                
            model_metrics = current["models"][model]
            model_metrics["calls"] += 1
            
            if metrics.get("success", False):
                model_metrics["successes"] += 1
            else:
                model_metrics["failures"] += 1
                
            model_metrics["total_tokens"] += tokens
            
            if "latency" in metrics:
                # Exponential moving average for latency
                alpha = 0.1  # Weight for new values
                model_metrics["average_latency"] = (alpha * metrics["latency"] + 
                                                 (1 - alpha) * model_metrics["average_latency"])
                                        
        # Save metrics periodically
        if current["calls"] % 10 == 0:
            await self._save_performance_metrics()
            
    async def generate_text(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          provider_name: Optional[str] = None,
                          model: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate text using the best available provider
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context
            provider_name: Optional specific provider to use
            model: Optional specific model to use
            **kwargs: Additional parameters
            
        Returns:
            Generated text and metadata
        """
        # Get the best provider
        provider = await self.get_provider(provider_name)
        
        if not provider:
            logger.warning("No LLM provider available")
            return {"error": "No LLM provider available"}
            
        # Start timing
        start_time = time.time()
        
        try:
            # Standardize parameters for this provider
            provider_specific_params = self._standardize_parameters(
                provider.get_provider_name(),
                model=model,
                **kwargs
            )
            
            # Generate text
            result = await provider.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                **provider_specific_params
            )
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            success = "text" in result and "error" not in result
            
            # Update metrics
            await self.update_metrics(provider.get_provider_name(), {
                "success": success,
                "tokens": result.get("tokens", 0),
                "latency": elapsed_time,
                "model": result.get("model", model)
            })
            
            return result
            
        except Exception as e:
            # Calculate metrics
            elapsed_time = time.time() - start_time
            
            # Update metrics for failure
            await self.update_metrics(provider.get_provider_name(), {
                "success": False,
                "tokens": 0,
                "latency": elapsed_time,
                "model": model
            })
            
            logger.error(f"Error generating text: {e}")
            return {"error": str(e)}
            
    def _standardize_parameters(self, provider_name: str, **kwargs) -> Dict[str, Any]:
        """
        Standardize parameters to provider-specific format
        
        Args:
            provider_name: Provider name
            **kwargs: Generic parameters
            
        Returns:
            Provider-specific parameters
        """
        if provider_name not in self.parameter_mappings:
            return kwargs
            
        mapping = self.parameter_mappings[provider_name]
        result = {}
        
        for param, value in kwargs.items():
            if param in mapping:
                result[mapping[param]] = value
            else:
                result[param] = value
                
        return result
            
    async def get_all_providers(self) -> Dict[str, Any]:
        """Get all registered providers with metrics"""
        result = {}
        
        # Add providers
        for name, provider in self.providers.items():
            result[name] = {
                "metrics": self.performance_metrics.get(name, {}),
                "capabilities": provider.get_capabilities(),
                "versions": self.provider_versions.get(name, []),
                "available": await provider.is_available()
            }
                
        return result
        
    async def _save_performance_metrics(self) -> None:
        """Save performance metrics to disk"""
        try:
            metrics_path = os.path.join(self.memory_path, "performance_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            
    def _load_performance_metrics(self) -> None:
        """Load performance metrics from disk"""
        try:
            metrics_path = os.path.join(self.memory_path, "performance_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            self.performance_metrics = {}
