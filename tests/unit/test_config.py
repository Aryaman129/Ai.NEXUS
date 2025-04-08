"""
Unit tests for the configuration module.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Config

class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        
        # Sample configuration
        self.sample_config = {
            "ui_config": {
                "theme": "light",
                "opacity": 0.8
            },
            "safety_config": {
                "safety_level": 1
            }
        }
        
        # Write sample configuration to the temporary file
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.sample_config, f)
            
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary file
        os.unlink(self.temp_file.name)
        
    def test_load_config(self):
        """Test loading configuration from a file."""
        config = Config(self.temp_file.name)
        self.assertEqual(config.config["ui_config"]["theme"], "light")
        self.assertEqual(config.config["ui_config"]["opacity"], 0.8)
        self.assertEqual(config.config["safety_config"]["safety_level"], 1)
        
    def test_default_config(self):
        """Test loading default configuration when file doesn't exist."""
        # Use a non-existent file
        config = Config("non_existent_file.json")
        # Check that default values are used
        self.assertIn("ui_config", config.config)
        self.assertIn("safety_config", config.config)
        
    def test_get_config_value(self):
        """Test getting configuration values."""
        config = Config(self.temp_file.name)
        # Test getting existing values
        self.assertEqual(config.get("ui_config.theme"), "light")
        self.assertEqual(config.get("ui_config.opacity"), 0.8)
        self.assertEqual(config.get("safety_config.safety_level"), 1)
        # Test getting non-existent values
        self.assertIsNone(config.get("non_existent_key"))
        self.assertEqual(config.get("non_existent_key", "default"), "default")
        
    def test_set_config_value(self):
        """Test setting configuration values."""
        config = Config(self.temp_file.name)
        # Test setting existing values
        config.set("ui_config.theme", "dark")
        self.assertEqual(config.get("ui_config.theme"), "dark")
        # Test setting new values
        config.set("new_key", "new_value")
        self.assertEqual(config.get("new_key"), "new_value")
        # Test setting nested values
        config.set("nested.key", "nested_value")
        self.assertEqual(config.get("nested.key"), "nested_value")
        
    def test_save_config(self):
        """Test saving configuration to a file."""
        config = Config(self.temp_file.name)
        # Modify the configuration
        config.set("ui_config.theme", "dark")
        # Save the configuration
        config.save_config()
        # Load the configuration again
        new_config = Config(self.temp_file.name)
        # Check that the changes were saved
        self.assertEqual(new_config.get("ui_config.theme"), "dark")
        
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_config_error(self, mock_json_dump, mock_file):
        """Test handling errors when saving configuration."""
        # Make json.dump raise an exception
        mock_json_dump.side_effect = Exception("Test exception")
        
        config = Config(self.temp_file.name)
        # Try to save the configuration
        result = config.save_config()
        # Check that the method returns False on error
        self.assertFalse(result)
        
if __name__ == "__main__":
    unittest.main()
