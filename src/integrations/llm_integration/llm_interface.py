"""
LLM Interface

This module defines the common interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Interface for all LLM providers"""
    
    @abstractmethod
    async def initialize(self, config: Optional[Dict] = None) -> bool:
        """Initialize the LLM provider with optional configuration"""
        pass
        
    @abstractmethod
    async def generate_text(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate text from the LLM
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt for context
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary containing:
            - text: The generated text
            - model: The model used
            - tokens: Token usage information
            - additional provider-specific information
        """
        pass
        
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from this provider
        
        Returns:
            List of dictionaries, each containing:
            - id: Model ID
            - name: Model name
            - context_length: Maximum context length
            - additional provider-specific information
        """
        pass
        
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the provider is available
        
        Returns:
            True if the provider is available, False otherwise
        """
        pass
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities and metadata"""
        return {
            "supports_streaming": False,
            "supports_system_prompt": True,
            "supports_function_calling": False,
            "supports_vision": False,
            "version": "0.1.0"
        }
        
    def get_provider_name(self) -> str:
        """Get the provider name"""
        return "base"
        
    def standardize_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Standardize parameters to provider-specific format
        
        Args:
            **kwargs: Generic parameters
            
        Returns:
            Provider-specific parameters
        """
        # Default implementation passes parameters through
        return kwargs
