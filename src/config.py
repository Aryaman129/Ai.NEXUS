"""
Configuration module for AI.NEXUS.

This module provides configuration settings for the AI.NEXUS system.
It loads settings from a JSON file and provides access to them through
a simple interface.
"""

import json
import os
import logging
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for AI.NEXUS."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default path.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config.json'
        )
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the JSON file.
        
        Returns:
            Dict containing configuration settings.
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Configuration file not found at {self.config_path}. Using default configuration.")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._default_config()
            
    def _default_config(self) -> Dict[str, Any]:
        """
        Provide default configuration settings.
        
        Returns:
            Dict containing default configuration settings.
        """
        return {
            "ui_config": {
                "theme": "dark",
                "opacity": 0.9,
                "width": 400,
                "height": 600
            },
            "safety_config": {
                "safety_level": 2,
                "require_confirmation_for_dangerous_actions": True
            },
            "monitor_interval": 1.0,
            "analysis_interval": 2.0,
            "max_visual_patterns": 10000,
            "visual_similarity_threshold": 0.7,
            "clarification_threshold": 0.7,
            "api_keys": {
                "hugging_face": "",
                "groq": "",
                "gemini": "",
                "together_ai": "",
                "mistral": "",
                "openrouter": ""
            },
            "llm_integration": {
                "default_provider": "groq",
                "task_preferences": {
                    "technical": {
                        "preferred_models": ["huggingface:mistralai/Mistral-7B-Instruct-v0.2", "ollama:deepseek-coder", "groq:mixtral-8x7b-32768"],
                        "temperature": 0.3,
                        "max_tokens": 2048
                    },
                    "creative": {
                        "preferred_models": ["ollama:llama2", "groq:llama2-70b-4096", "gemini:gemini-pro"],
                        "temperature": 0.7,
                        "max_tokens": 1024
                    },
                    "factual": {
                        "preferred_models": ["groq:mixtral-8x7b-32768", "ollama:deepseek-coder", "mistral:mistral-medium"],
                        "temperature": 0.1,
                        "max_tokens": 1024
                    },
                    "general": {
                        "preferred_models": ["ollama:llama2", "groq:mixtral-8x7b-32768", "gemini:gemini-pro"],
                        "temperature": 0.5,
                        "max_tokens": 1024
                    }
                }
            },
            "ui_detection": {
                "default_detector": "autogluon",
                "confidence_threshold": 0.6,
                "detectors": {
                    "autogluon": {
                        "priority": 100,
                        "model_path": "models/autogluon_ui_detector"
                    },
                    "yolo": {
                        "priority": 80,
                        "model_path": "models/yolov8n_ui.pt"
                    },
                    "huggingface": {
                        "priority": 60,
                        "model_id": "facebook/detr-resnet-50"
                    },
                    "opencv": {
                        "priority": 10
                    }
                }
            }
        }
        
    def save_config(self) -> bool:
        """
        Save the current configuration to the JSON file.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key to get.
            default: The default value to return if the key is not found.
            
        Returns:
            The configuration value, or the default if not found.
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key to set.
            value: The value to set.
            
        Returns:
            True if successful, False otherwise.
        """
        keys = key.split('.')
        config = self.config
        
        try:
            # Navigate to the nested dictionary
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
                
            # Set the value
            config[keys[-1]] = value
            
            # Save the configuration
            return self.save_config()
        except Exception as e:
            logger.error(f"Error setting configuration: {e}")
            return False
            
# Create a singleton instance
config = Config()

def get_config() -> Config:
    """
    Get the configuration instance.
    
    Returns:
        The configuration instance.
    """
    return config
