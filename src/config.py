"""
NEXUS Configuration
Centralized configuration for NEXUS components and services
"""
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class NexusConfig:
    """Configuration manager for NEXUS"""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "general": {
            "debug_mode": False,
            "save_history": True,
            "history_file": "nexus_history.json",
            "memory_file": "nexus_memory.json"
        },
        "vision": {
            "save_screenshots": False,
            "screenshot_dir": "screenshots",
            "max_screenshots": 100,
            "screenshot_interval": 1.0,
            "ocr": {
                "enabled": True,
                "engine": "tesseract"
            },
            "object_detection": {
                "enabled": False,
                "model": "yolov5",
                "confidence": 0.5
            }
        },
        "ai": {
            "default_llm": "ollama",
            "ollama": {
                "model": "llama3",
                "host": "http://localhost:11434",
                "timeout": 60
            },
            "summarization": {
                "use_ai_models": True,
                "model": "facebook/bart-large-cnn",
                "fallback_model": "sshleifer/distilbart-cnn-12-6"
            }
        },
        "research": {
            "max_search_results": 5,
            "extract_content": True,
            "max_content_length": 10000
        },
        "ui_automation": {
            "click_delay": 0.5,
            "typing_speed": 0.05,
            "safe_mode": True
        },
        "tools": {
            "auto_discover": True,
            "enabled_categories": ["all"]
        }
    }
    
    def __init__(self, config_path=None):
        """Initialize configuration with optional path to config file"""
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.config_dir / "nexus_config.json"
            
        # Load or create config
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                
                # Update with any missing default values
                return self._update_with_defaults(config)
            else:
                logger.info("No configuration file found, using defaults")
                self._save_config(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self.DEFAULT_CONFIG
            
    def _update_with_defaults(self, config):
        """Update configuration with any missing default values"""
        updated_config = self.DEFAULT_CONFIG.copy()
        
        # Recursively update configuration
        def update_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    update_dict(target[key], value)
                else:
                    target[key] = value
                    
        update_dict(updated_config, config)
        return updated_config
        
    def _save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def get(self, section, key=None):
        """Get configuration value(s)"""
        try:
            if section in self.config:
                if key is None:
                    return self.config[section]
                elif key in self.config[section]:
                    return self.config[section][key]
            return None
        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return None
            
    def set(self, section, key, value):
        """Set configuration value"""
        try:
            if section not in self.config:
                self.config[section] = {}
                
            self.config[section][key] = value
            self._save_config()
            return True
        except Exception as e:
            logger.error(f"Error setting configuration: {e}")
            return False
            
    def reset(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        self._save_config()
        logger.info("Reset configuration to defaults")
        
    def get_all(self):
        """Get entire configuration"""
        return self.config
