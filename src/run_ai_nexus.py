"""
Main entry point for the AI.NEXUS application.

This module initializes and runs the AI.NEXUS application.
"""

import argparse
import logging
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the AI.NEXUS application.')
    parser.add_argument('--config', type=str, help='Path to the configuration file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load configuration
    config = get_config()
    if args.config:
        config.config_path = args.config
        config._load_config()
        
    logger.info("Starting AI.NEXUS...")
    
    # TODO: Initialize and run the application
    # This will be implemented as the project progresses
    
    logger.info("AI.NEXUS is running. Press Ctrl+C to exit.")
    
    try:
        # Keep the application running
        # This will be replaced with the actual application loop
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down AI.NEXUS...")
        # TODO: Perform cleanup
        
    logger.info("AI.NEXUS has been shut down.")

if __name__ == "__main__":
    main()
