#!/usr/bin/env python3
"""
NEXUS - Autonomous AI Orchestration System
Run Script
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Make sure the directory structure is properly set
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "templates",
        os.path.join("templates", "search_icon.png"),
    ]
    
    for directory in directories:
        path = Path(directory)
        if not os.path.exists(directory) and not path.suffix:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

async def main():
    """Main entry point for NEXUS"""
    try:
        # Ensure required directories exist
        ensure_directories()
        
        # Import the NEXUS core module
        from nexus_core import NexusAutonomousDemo
        
        logger.info("Starting NEXUS Autonomous Demo")
        nexus = NexusAutonomousDemo()
        await nexus.run_demo()
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please make sure you have installed all requirements with 'pip install -r requirements.txt'")
    except Exception as e:
        logger.error(f"Error running NEXUS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 50)
    print(" NEXUS: Autonomous AI Orchestration System")
    print("=" * 50)
    print("\nInitializing the AI-driven orchestration system...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNEXUS terminated by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
