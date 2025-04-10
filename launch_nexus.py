"""
NEXUS Adaptive System Launcher

This script launches the NEXUS adaptive automation system with an always-on-top UI,
allowing it to monitor and control multiple applications while learning from interactions.
"""
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nexus.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NEXUS")

def setup_environment():
    """Set up the NEXUS environment"""
    # Create necessary directories
    os.makedirs("memory", exist_ok=True)
    os.makedirs("memory/applications", exist_ok=True)
    os.makedirs("memory/applications/visual_memory", exist_ok=True)
    os.makedirs("memory/applications/clarification_memory", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check for required packages
    try:
        import pyautogui
        import pygetwindow
        import numpy
        import cv2
        import tkinter
        import PIL
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Installing required packages...")
        
        os.system("pip install pyautogui pygetwindow numpy opencv-python pillow")
        
        logger.info("Packages installed. Restarting script...")
        # Restart the script
        os.execv(sys.executable, ['python'] + sys.argv)

def load_config(config_path=None):
    """Load configuration from file or create default"""
    default_config = {
        "ui_config": {
            "theme": "dark",
            "opacity": 0.9,
            "width": 400,
            "height": 600
        },
        "safety_config": {
            "safety_level": 2,  # 1-5, higher is more restrictive
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
            "together_ai": "4ec34405c082ae11d558aabe290486bd73ae6897fb623ba0bba481df21f5ec39"
        }
    }
    
    # If config path provided, try to load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                
            # Merge with default config
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
                    
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info("Using default configuration")
        
        # Save default config for future reference
        with open("nexus_config.json", 'w') as f:
            json.dump(default_config, f, indent=4)
            
        logger.info("Default configuration saved to nexus_config.json")
    
    return default_config

def launch_nexus(config, mode="ui"):
    """Launch NEXUS with the specified mode"""
    if mode == "ui":
        # Launch with UI and multi-app controller
        from src.nexus_ui.multi_app_controller import MultiAppController
        
        logger.info("Launching NEXUS with UI and multi-app controller")
        
        controller = MultiAppController(config=config)
        
        # Calculate position (right side of screen)
        import pyautogui
        screen_width, screen_height = pyautogui.size()
        position = (screen_width - 420, 50)
        
        # Start controller
        controller.start(position)
        
        try:
            # Keep main thread alive
            while controller.ui.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            # Stop on Ctrl+C
            logger.info("Received keyboard interrupt, stopping NEXUS")
            controller.stop()
    
    elif mode == "headless":
        # Launch without UI (for scripting)
        logger.info("Launching NEXUS in headless mode")
        # This would integrate with your existing NEXUS core
        # Implementation depends on your NEXUS core architecture
        raise NotImplementedError("Headless mode not implemented yet")
    
    else:
        logger.error(f"Unknown launch mode: {mode}")

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch NEXUS adaptive automation system")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", choices=["ui", "headless"], default="ui", 
                      help="Launch mode (ui or headless)")
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    # Display header
    print(r"""
    _   _ _______  ___   _ ____  
    | \ | | ____\ \/ / | | / ___| 
    |  \| |  _|  \  /| | | \___ \ 
    | |\  | |___ /  \| |_| |___) |
    |_| \_|_____/_/\_\\___/|____/ 
                                
    Adaptive Automation System
    """)
    
    print("Starting NEXUS adaptive automation system...")
    print(f"Launch mode: {args.mode}")
    print("Type 'help' in the NEXUS UI for available commands")
    print("-" * 50)
    
    # Launch NEXUS
    launch_nexus(config, args.mode)
    
    print("NEXUS has been shut down")

if __name__ == "__main__":
    main()
