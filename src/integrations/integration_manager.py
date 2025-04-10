"""
Integration Manager for NEXUS
Handles dynamic discovery, registration, and orchestration of various AI services
"""
import os
import logging
import importlib
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
import inspect

logger = logging.getLogger(__name__)

class IntegrationManager:
    """
    Dynamically manages integrations with various AI services and APIs.
    
    This class is central to NEXUS's adaptive architecture, allowing
    the system to discover, register, and orchestrate different AI services
    based on availability and task requirements.
    """
    
    def __init__(self):
        """Initialize the integration manager"""
        # Dictionary to store available integrations
        self.integrations = {}
        
        # Dictionary to store registered capabilities
        self.capabilities = {}
        
        # Register of capability providers (which integration can provide what capability)
        self.capability_providers = {}
        
        # Performance tracking for adaptive selection
        self.performance_metrics = {}
        
        # Detect and register available integrations
        self._discover_integrations()
        
    def _discover_integrations(self):
        """Automatically discover and initialize available integrations"""
        # List of potential integrations to check
        potential_integrations = [
            # LLM Services
            {"name": "gemini", "module": "gemini_integration", "class": "GeminiIntegration"},
            {"name": "groq", "module": "groq_integration", "class": "GroqIntegration"},
            {"name": "huggingface", "module": "huggingface_integration", "class": "HuggingFaceIntegration"},
            
            # Vision Services
            {"name": "adaptive_vision", "module": "adaptive_vision", "class": "AdaptiveVision"},
            {"name": "visual_intelligence", "module": "nexus_visual_intelligence", "class": "NexusVisualIntelligence"},
            
            # Search Services
            {"name": "duckduckgo", "module": "duckduckgo_search", "class": "DuckDuckGoSearch"},
            {"name": "multimodal_search", "module": "multimodal_search", "class": "MultimodalSearch"},
            
            # Vector Storage
            {"name": "vector_wrapper", "module": "async_vector_wrapper", "class": "AsyncVectorWrapper"},
        ]
        
        # Try to initialize each integration
        for integration in potential_integrations:
            try:
                # Dynamically import the module
                module_path = f"src.integrations.{integration['module']}"
                module = importlib.import_module(module_path)
                
                # Get the class
                integration_class = getattr(module, integration["class"])
                
                # Initialize the integration
                integration_instance = integration_class()
                
                # Register the integration
                self.integrations[integration["name"]] = integration_instance
                
                # Register the integration's capabilities
                self._register_capabilities(integration["name"], integration_instance)
                
                logger.info(f"Successfully registered integration: {integration['name']}")
            except (ImportError, AttributeError, Exception) as e:
                logger.warning(f"Failed to load integration {integration['name']}: {e}")
    
    def _register_capabilities(self, integration_name: str, integration_instance: Any):
        """Register the capabilities provided by an integration"""
        # Check for common capability methods
        capabilities = {
            "text_generation": hasattr(integration_instance, "generate_text") or 
                             hasattr(integration_instance, "generate_content") or
                             hasattr(integration_instance, "generate"),
                             
            "image_analysis": hasattr(integration_instance, "analyze_image") or
                            hasattr(integration_instance, "vision_analysis") or
                            hasattr(integration_instance, "process_image"),
                            
            "text_embedding": hasattr(integration_instance, "get_embeddings") or
                            hasattr(integration_instance, "embed_text"),
                            
            "web_search": hasattr(integration_instance, "search") or
                        hasattr(integration_instance, "web_search"),
                        
            "document_processing": hasattr(integration_instance, "process_document") or
                                 hasattr(integration_instance, "extract_text"),
        }
        
        # Register each capability
        for capability, has_capability in capabilities.items():
            if has_capability:
                if capability not in self.capability_providers:
                    self.capability_providers[capability] = []
                    
                self.capability_providers[capability].append(integration_name)
                logger.info(f"Registered capability '{capability}' from {integration_name}")
    
    async def get_best_provider(self, capability: str, context: Dict[str, Any] = None) -> str:
        """
        Get the best provider for a specific capability based on context and performance
        
        Args:
            capability: The capability needed
            context: Optional context that might influence the selection
            
        Returns:
            Name of the best provider for this capability
        """
        if capability not in self.capability_providers or not self.capability_providers[capability]:
            return None
            
        # If only one provider is available, use it
        if len(self.capability_providers[capability]) == 1:
            return self.capability_providers[capability][0]
            
        # Consider context for specific capability types
        if context:
            # For image analysis with high quality needs, prefer certain providers
            if capability == "image_analysis" and context.get("quality_required") == "high":
                for provider in ["adaptive_vision", "huggingface"]:
                    if provider in self.capability_providers[capability]:
                        return provider
            
            # For text generation with specific model preference
            if capability == "text_generation" and "model_preference" in context:
                model_preference = context["model_preference"]
                
                # Check which provider can handle this model
                for provider in self.capability_providers[capability]:
                    integration = self.integrations[provider]
                    
                    # Different integrations might have different methods to check model availability
                    if hasattr(integration, "is_model_available"):
                        if await integration.is_model_available(model_preference):
                            return provider
                    elif hasattr(integration, "list_models") and callable(integration.list_models):
                        models = integration.list_models()
                        if model_preference in models:
                            return provider
        
        # Consider performance metrics if available
        best_provider = None
        best_score = float("-inf")
        
        for provider in self.capability_providers[capability]:
            if provider in self.performance_metrics and capability in self.performance_metrics[provider]:
                metrics = self.performance_metrics[provider][capability]
                
                # Calculate a score based on success rate and response time
                success_rate = metrics.get("success_rate", 0)
                avg_response_time = metrics.get("avg_response_time", float("inf"))
                
                # Higher success rate and lower response time is better
                if avg_response_time > 0:
                    time_factor = 1.0 / avg_response_time
                else:
                    time_factor = 1.0
                    
                score = (success_rate * 0.7) + (time_factor * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_provider = provider
        
        # If we found a provider based on metrics, return it
        if best_provider:
            return best_provider
            
        # Default to first available provider
        return self.capability_providers[capability][0]
    
    async def execute_capability(self, 
                               capability: str, 
                               provider: str = None, 
                               **kwargs) -> Any:
        """
        Execute a capability with a specific provider or the best available one
        
        Args:
            capability: The capability to execute
            provider: Optional specific provider to use
            **kwargs: Arguments to pass to the capability method
            
        Returns:
            Result of the capability execution
        """
        # Get the best provider if none specified
        if not provider:
            provider = await self.get_best_provider(capability, kwargs.get("context"))
            
        if not provider or provider not in self.integrations:
            return {"error": f"No provider available for capability: {capability}"}
            
        integration = self.integrations[provider]
        
        # Track start time for performance metrics
        start_time = asyncio.get_event_loop().time()
        success = False
        
        try:
            # Map capability to method names
            capability_methods = {
                "text_generation": ["generate_text", "generate_content", "generate"],
                "image_analysis": ["analyze_image", "vision_analysis", "process_image"],
                "text_embedding": ["get_embeddings", "embed_text"],
                "web_search": ["search", "web_search"],
                "document_processing": ["process_document", "extract_text"],
            }
            
            # Find the appropriate method
            method_name = None
            for potential_method in capability_methods.get(capability, []):
                if hasattr(integration, potential_method) and callable(getattr(integration, potential_method)):
                    method_name = potential_method
                    break
                    
            if not method_name:
                return {"error": f"Provider {provider} does not support capability {capability}"}
                
            # Get the method
            method = getattr(integration, method_name)
            
            # Check if the method is async
            if inspect.iscoroutinefunction(method):
                result = await method(**kwargs)
            else:
                # Execute in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: method(**kwargs))
                
            success = "error" not in result if isinstance(result, dict) else True
            return result
            
        except Exception as e:
            logger.error(f"Error executing capability {capability} with provider {provider}: {e}")
            return {"error": str(e)}
            
        finally:
            # Update performance metrics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self._update_performance_metrics(provider, capability, elapsed_time, success)
    
    def _update_performance_metrics(self, 
                                  provider: str, 
                                  capability: str, 
                                  response_time: float, 
                                  success: bool):
        """Update performance metrics for a provider and capability"""
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = {}
            
        if capability not in self.performance_metrics[provider]:
            self.performance_metrics[provider][capability] = {
                "success_count": 0,
                "total_count": 0,
                "total_time": 0,
                "success_rate": 0,
                "avg_response_time": 0
            }
            
        metrics = self.performance_metrics[provider][capability]
        
        # Update counts
        if success:
            metrics["success_count"] += 1
        metrics["total_count"] += 1
        metrics["total_time"] += response_time
        
        # Update rates
        metrics["success_rate"] = metrics["success_count"] / metrics["total_count"]
        metrics["avg_response_time"] = metrics["total_time"] / metrics["total_count"]
    
    def get_capability_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all capabilities and providers"""
        return self.performance_metrics
        
    def register_custom_integration(self, name: str, integration_instance: Any):
        """
        Register a custom integration manually
        
        Args:
            name: Name for the integration
            integration_instance: Instance of the integration
        """
        self.integrations[name] = integration_instance
        self._register_capabilities(name, integration_instance)
        
    def create_pipeline(self, pipeline_config: Dict[str, Any]) -> Any:
        """
        Create a pipeline by combining multiple capabilities
        
        A pipeline is a sequence of capabilities that are executed in order,
        with the output of one capability feeding into the next.
        
        Args:
            pipeline_config: Configuration for the pipeline
            
        Returns:
            A pipeline object that can be executed
        """
        # Implementation details would depend on specific pipeline needs
        # This is a placeholder for the concept
        return {"pipeline_config": pipeline_config, "status": "created"}
        
    async def execute_pipeline(self, pipeline: Any, input_data: Any) -> Any:
        """
        Execute a pipeline with the given input data
        
        Args:
            pipeline: Pipeline object created by create_pipeline
            input_data: Input data for the pipeline
            
        Returns:
            Result of pipeline execution
        """
        # Implementation details would depend on specific pipeline needs
        # This is a placeholder for the concept
        return {"pipeline": pipeline, "input": input_data, "status": "executed"}
