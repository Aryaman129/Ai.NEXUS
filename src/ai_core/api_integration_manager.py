"""
API Integration Manager for NEXUS

This module implements the core integration manager for all external API services,
coordinating their usage, adaptive selection, and knowledge distillation for continuous learning.

Key features:
- Unified interface to multiple API providers
- Adaptive service selection based on task requirements and past performance
- Rate limit management and API key rotation
- Knowledge distillation to enhance local models using API responses
- Transparent fallback mechanisms for robustness
"""
import os
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

class APIIntegrationManager:
    """
    Unified manager for all API integrations in NEXUS
    
    This module embodies NEXUS's core philosophy of adaptive learning
    by tracking performance metrics, managing rate limits, and enabling
    knowledge distillation between service providers.
    """
    
    def __init__(self, shared_memory=None, vector_storage=None):
        """
        Initialize the API Integration Manager
        
        Args:
            shared_memory: Shared memory system for experience tracking
            vector_storage: Vector storage for knowledge distillation
        """
        self.providers = {}
        self.capabilities = {}
        self.performance_metrics = {}
        self.shared_memory = shared_memory
        self.vector_storage = vector_storage
        self.rate_limit_manager = RateLimitManager()
        self.key_manager = APIKeyManager()
        self.learning_buffer = []
        
        # Path for storing performance metrics
        self.metrics_dir = Path("memory/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / "api_performance.json"
        
        # Load existing metrics if available
        self._load_performance_metrics()
        
    async def initialize_providers(self):
        """
        Initialize all available API providers
        
        This method discovers and registers all API service providers,
        checks their availability, and builds the capability map.
        
        Returns:
            Dictionary of available providers
        """
        from src.integrations.openrouter_integration import OpenRouterIntegration
        from src.integrations.mistral_integration import MistralAIIntegration
        from src.integrations.huggingface_integration import HuggingFaceIntegration
        
        try:
            # Import optional integrations if installed
            from src.integrations.groq_integration import GroqIntegration
            groq_available = True
        except ImportError:
            groq_available = False
            
        try:
            from src.integrations.google_ai_integration import GoogleAIIntegration
            google_ai_available = True
        except ImportError:
            google_ai_available = False
        
        # Initialize providers
        integration_classes = {
            "openrouter": OpenRouterIntegration,
            "mistral": MistralAIIntegration,
            "huggingface": HuggingFaceIntegration
        }
        
        if groq_available:
            integration_classes["groq"] = GroqIntegration
            
        if google_ai_available:
            integration_classes["google"] = GoogleAIIntegration
            
        # Initialize each provider
        for name, IntegrationClass in integration_classes.items():
            try:
                # Get API key from key manager
                api_key = self.key_manager.get_next_available_key(name)
                
                # Create integration instance
                integration = IntegrationClass(api_key=api_key)
                
                # Initialize the integration
                available = await integration.initialize()
                
                if available:
                    # Register provider
                    self.providers[name] = integration
                    logger.info(f"Registered {name} integration")
                    
                    # Register capabilities
                    if hasattr(integration, 'get_capabilities'):
                        capabilities = await integration.get_capabilities()
                    else:
                        capabilities = ["text_generation"]  # Default capability
                        
                    for capability in capabilities:
                        if capability not in self.capabilities:
                            self.capabilities[capability] = []
                        self.capabilities[capability].append(name)
                else:
                    logger.warning(f"{name} integration not available")
            except Exception as e:
                logger.error(f"Error initializing {name} integration: {e}")
                
        return self.providers
    
    async def execute(self, capability: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a capability using the best available provider
        
        This method implements NEXUS's adaptive execution by selecting
        the optimal provider based on task requirements and past performance.
        
        Args:
            capability: The capability to execute (e.g., "text_generation")
            **kwargs: Parameters for the capability
            
        Returns:
            Result dictionary with provider metadata
        """
        # Check if we have providers for this capability
        if capability not in self.capabilities or not self.capabilities[capability]:
            logger.warning(f"No providers available for capability: {capability}")
            return {"error": f"No providers available for capability: {capability}"}
            
        # Get context and task type
        context = kwargs.get('context', {})
        task_type = context.get('task_type', 'general')
        
        # Try to use preferred provider if specified
        preferred_provider = kwargs.get('preferred_provider')
        if preferred_provider and preferred_provider in self.providers:
            if preferred_provider in self.capabilities[capability]:
                result = await self._execute_with_provider(
                    provider=preferred_provider,
                    capability=capability,
                    **kwargs
                )
                if 'error' not in result:
                    return result
        
        # Select best provider based on capability and performance history
        provider = await self._select_best_provider(capability, task_type)
        
        # Execute with selected provider
        result = await self._execute_with_provider(
            provider=provider,
            capability=capability, 
            **kwargs
        )
        
        # If execution failed, try fallback providers
        if 'error' in result:
            fallback_result = await self._execute_with_fallbacks(
                capability=capability,
                failed_provider=provider,
                **kwargs
            )
            if 'error' not in fallback_result:
                return fallback_result
                
        # Record result for knowledge distillation if successful
        if 'error' not in result and self.shared_memory:
            await self._record_for_knowledge_distillation(
                capability=capability,
                prompt=kwargs.get('prompt', ''),
                result=result,
                provider=provider,
                task_type=task_type
            )
            
        return result
    
    async def _execute_with_provider(self, provider: str, capability: str, **kwargs) -> Dict[str, Any]:
        """Execute a capability with a specific provider"""
        if provider not in self.providers:
            return {"error": f"Provider {provider} not available"}
            
        try:
            # Check rate limits
            await self.rate_limit_manager.wait_if_needed(provider)
            
            # Track start time for performance metrics
            start_time = time.time()
            
            # Execute the capability
            if hasattr(self.providers[provider], 'execute'):
                result = await self.providers[provider].execute(capability, **kwargs)
            else:
                # Fallback if execute method doesn't exist
                method = getattr(self.providers[provider], capability, None)
                if method:
                    result = await method(**kwargs)
                else:
                    return {"error": f"Provider {provider} does not support capability {capability}"}
                    
            # Track performance
            elapsed_time = time.time() - start_time
            success = 'error' not in result
            
            # Update rate limit manager
            if not success and "rate limit" in str(result.get('text', '')).lower():
                self.rate_limit_manager.mark_rate_limited(provider)
                
            # Update performance metrics
            self._update_performance_metrics(
                provider=provider,
                capability=capability,
                elapsed_time=elapsed_time,
                success=success,
                task_type=kwargs.get('context', {}).get('task_type', 'general')
            )
            
            # Add provider info to result
            result['provider'] = provider
            return result
            
        except Exception as e:
            logger.error(f"Error executing {capability} with {provider}: {e}")
            return {"error": str(e), "provider": provider}
    
    async def _execute_with_fallbacks(self, capability: str, failed_provider: str, **kwargs) -> Dict[str, Any]:
        """Try fallback providers if the primary provider fails"""
        providers = self.capabilities.get(capability, [])
        
        # Filter out the failed provider
        fallbacks = [p for p in providers if p != failed_provider]
        
        # Sort fallbacks by performance
        fallbacks.sort(
            key=lambda p: self._get_provider_score(p, capability, kwargs.get('context', {}).get('task_type')),
            reverse=True
        )
        
        # Try each fallback
        for provider in fallbacks:
            result = await self._execute_with_provider(
                provider=provider,
                capability=capability,
                **kwargs
            )
            
            if 'error' not in result:
                result['fallback'] = True
                return result
                
        # If all fallbacks fail, return the original error
        return {"error": "All providers failed", "capability": capability}
    
    async def _select_best_provider(self, capability: str, task_type: str = 'general') -> str:
        """
        Select the best provider for a capability based on performance history
        
        This implements NEXUS's adaptive learning by selecting providers
        based on their historical performance for specific tasks.
        """
        providers = self.capabilities.get(capability, [])
        
        if not providers:
            return None
            
        # Calculate scores for each provider
        scores = {}
        for provider in providers:
            if not self.rate_limit_manager.is_available(provider):
                # Skip rate-limited providers
                continue
                
            scores[provider] = self._get_provider_score(provider, capability, task_type)
            
        # Return the provider with the highest score, or first available if no scores
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Use any non-rate-limited provider
            for provider in providers:
                if self.rate_limit_manager.is_available(provider):
                    return provider
                    
        # Last resort - use the first provider even if rate-limited
        return providers[0]
    
    def _get_provider_score(self, provider: str, capability: str, task_type: str = 'general') -> float:
        """Calculate a score for a provider based on performance metrics"""
        # Default score for providers with no data
        base_score = 0.5
        
        # Get metrics for this provider and capability
        key = f"{provider}:{capability}"
        if key not in self.performance_metrics:
            return base_score
            
        metrics = self.performance_metrics[key]
        
        # Calculate success rate component (60% weight)
        success_rate = metrics.get('success_rate', 0.5)
        success_component = success_rate * 0.6
        
        # Calculate speed component (30% weight)
        avg_time = metrics.get('avg_time', 1.0)
        # Normalize to 0-1 range where 1 is fast (0.1s) and 0 is slow (5s+)
        speed = max(0, min(1, 1 - (avg_time / 5.0)))
        speed_component = speed * 0.3
        
        # Calculate task-specific component (10% weight)
        task_component = 0.0
        task_metrics = metrics.get('tasks', {}).get(task_type, {})
        if task_metrics:
            task_success_rate = task_metrics.get('success_rate', 0.5)
            task_component = task_success_rate * 0.1
            
        # Return total score
        return success_component + speed_component + task_component
    
    def _update_performance_metrics(self, provider: str, capability: str, elapsed_time: float, 
                                   success: bool, task_type: str = 'general'):
        """Update performance metrics for a provider"""
        # Create key for this provider-capability pair
        key = f"{provider}:{capability}"
        
        # Initialize metrics if needed
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                'success_count': 0,
                'failure_count': 0,
                'total_time': 0.0,
                'call_count': 0,
                'avg_time': 0.0,
                'success_rate': 0.0,
                'tasks': {}
            }
            
        metrics = self.performance_metrics[key]
        
        # Update general metrics
        metrics['call_count'] += 1
        metrics['total_time'] += elapsed_time
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['failure_count'] += 1
            
        # Calculate averages
        metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
        metrics['success_rate'] = metrics['success_count'] / metrics['call_count']
        
        # Update task-specific metrics
        if task_type:
            if task_type not in metrics['tasks']:
                metrics['tasks'][task_type] = {
                    'success_count': 0,
                    'failure_count': 0,
                    'call_count': 0,
                    'success_rate': 0.0
                }
                
            task_metrics = metrics['tasks'][task_type]
            task_metrics['call_count'] += 1
            
            if success:
                task_metrics['success_count'] += 1
            else:
                task_metrics['failure_count'] += 1
                
            task_metrics['success_rate'] = task_metrics['success_count'] / task_metrics['call_count']
            
        # Save metrics periodically
        if metrics['call_count'] % 10 == 0:
            self._save_performance_metrics()
            
    def _load_performance_metrics(self):
        """Load performance metrics from disk"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
                logger.info(f"Loaded performance metrics for {len(self.performance_metrics)} provider-capability pairs")
            except Exception as e:
                logger.error(f"Error loading performance metrics: {e}")
                self.performance_metrics = {}
        else:
            logger.info("No performance metrics file found, starting fresh")
            self.performance_metrics = {}
            
    def _save_performance_metrics(self):
        """Save performance metrics to disk"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            logger.info(f"Saved performance metrics for {len(self.performance_metrics)} provider-capability pairs")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            
    async def _record_for_knowledge_distillation(self, capability: str, prompt: str, 
                                               result: Dict[str, Any], provider: str, task_type: str):
        """Record successful API responses for knowledge distillation"""
        # Only process successful, high-quality responses
        if 'error' in result or not result.get('text'):
            return
            
        # Add to learning buffer
        self.learning_buffer.append({
            'capability': capability,
            'prompt': prompt,
            'response': result.get('text', ''),
            'provider': provider,
            'model': result.get('model', 'unknown'),
            'task_type': task_type,
            'timestamp': time.time()
        })
        
        # When buffer reaches threshold, process for knowledge distillation
        if len(self.learning_buffer) >= 10:
            await self._process_learning_buffer()
            
    async def _process_learning_buffer(self):
        """Process the learning buffer for knowledge distillation"""
        if not self.learning_buffer:
            return
            
        if self.vector_storage:
            # Add examples to vector storage for RAG
            texts = [f"PROMPT: {item['prompt']}\nRESPONSE: {item['response']}" 
                    for item in self.learning_buffer]
            
            metadatas = [
                {
                    "source": "api_learning",
                    "provider": item["provider"],
                    "model": item["model"],
                    "task_type": item["task_type"],
                    "timestamp": item["timestamp"]
                }
                for item in self.learning_buffer
            ]
            
            self.vector_storage.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} examples to vector storage for knowledge distillation")
            
        if self.shared_memory:
            # Add examples to shared memory for experience recall
            for item in self.learning_buffer:
                self.shared_memory.add_experience(
                    task=item["task_type"],
                    input_data=item["prompt"],
                    output_data=item["response"],
                    metadata={
                        "source": "api_learning",
                        "provider": item["provider"],
                        "model": item["model"],
                        "confidence": 0.95
                    }
                )
            logger.info(f"Added {len(self.learning_buffer)} experiences to shared memory")
            
        # Clear the buffer
        self.learning_buffer = []


class RateLimitManager:
    """Manages API rate limits to prevent errors"""
    def __init__(self):
        self.rate_limits = {
            "openrouter": {"requests_per_second": 1.0, "last_request": 0, "rate_limited": False},
            "mistral": {"requests_per_second": 0.5, "last_request": 0, "rate_limited": False},
            "huggingface": {"requests_per_second": 1.0, "last_request": 0, "rate_limited": False},
            "groq": {"requests_per_second": 5.0, "last_request": 0, "rate_limited": False},
            "google": {"requests_per_second": 10.0, "last_request": 0, "rate_limited": False}
        }
        
    async def wait_if_needed(self, provider: str):
        """Wait if necessary to avoid rate limit"""
        if provider not in self.rate_limits:
            return
            
        limit = self.rate_limits[provider]
        min_interval = 1.0 / limit["requests_per_second"]
        
        # Check time since last request
        elapsed = time.time() - limit["last_request"]
        if elapsed < min_interval:
            # Wait remaining time
            wait_time = min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before next request to {provider}")
            await asyncio.sleep(wait_time)
            
        # Update last request time
        self.rate_limits[provider]["last_request"] = time.time()
        
    def mark_rate_limited(self, provider: str):
        """Mark a provider as rate-limited"""
        if provider not in self.rate_limits:
            return
            
        self.rate_limits[provider]["rate_limited"] = True
        self.rate_limits[provider]["limited_at"] = time.time()
        
        # Increase the delay between requests
        self.rate_limits[provider]["requests_per_second"] *= 0.5
        
        logger.warning(f"Provider {provider} rate limited. Adjusted to " +
                      f"{self.rate_limits[provider]['requests_per_second']} req/s")
        
        # Schedule reset after cooldown
        asyncio.create_task(self._reset_rate_limit(provider, cooldown=60))
        
    async def _reset_rate_limit(self, provider: str, cooldown: int = 60):
        """Reset rate limit status after cooldown period"""
        await asyncio.sleep(cooldown)
        if provider in self.rate_limits:
            self.rate_limits[provider]["rate_limited"] = False
            logger.info(f"Provider {provider} rate limit status reset after cooldown")
            
    def is_available(self, provider: str) -> bool:
        """Check if a provider is currently available (not rate-limited)"""
        if provider not in self.rate_limits:
            return True
            
        return not self.rate_limits[provider].get("rate_limited", False)


class APIKeyManager:
    """Manages multiple API keys for each provider"""
    def __init__(self):
        self.api_keys = {
            "openrouter": [
                os.environ.get("OPENROUTER_API_KEY", "")
            ],
            "mistral": [
                os.environ.get("MISTRAL_API_KEY", "")
            ],
            "huggingface": [
                os.environ.get("HF_API_KEY", "")
            ],
            "groq": [
                os.environ.get("GROQ_API_KEY", "")
            ],
            "google": [
                os.environ.get("GOOGLE_AI_API_KEY", "")
            ]
        }
        
        # Remove empty keys
        for provider in list(self.api_keys.keys()):
            self.api_keys[provider] = [key for key in self.api_keys[provider] if key]
            
        self.key_status = {}  # Track which keys are working/rate-limited
        
    def get_next_available_key(self, provider: str) -> Optional[str]:
        """Get the next available API key for a provider"""
        if provider not in self.api_keys or not self.api_keys[provider]:
            return None
            
        # Rotate through keys, prioritizing those not rate-limited
        for key in self.api_keys[provider]:
            status = self.key_status.get((provider, key), {"available": True})
            if status["available"]:
                return key
                
        # If all keys are rate-limited, use the least recently used one
        return self.api_keys[provider][0]
        
    def mark_key_rate_limited(self, provider: str, key: str):
        """Mark a key as rate-limited"""
        self.key_status[(provider, key)] = {
            "available": False,
            "limited_at": time.time()
        }
        
        # Schedule key to become available again after cooldown
        asyncio.create_task(self._reset_key_after_cooldown(provider, key))
        
    async def _reset_key_after_cooldown(self, provider: str, key: str, cooldown: int = 60):
        """Reset key availability after cooldown period"""
        await asyncio.sleep(cooldown)
        self.key_status[(provider, key)] = {"available": True}
        
    def add_api_key(self, provider: str, key: str):
        """Add a new API key for a provider"""
        if provider not in self.api_keys:
            self.api_keys[provider] = []
            
        if key and key not in self.api_keys[provider]:
            self.api_keys[provider].append(key)
            return True
        return False
