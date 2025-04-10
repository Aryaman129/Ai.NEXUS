"""
NEXUS Visual Intelligence Demo
Showcases the combined power of Google Cloud services and adaptive learning
"""
import asyncio
import os
import sys
import time
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add NEXUS to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.integrations.nexus_visual_intelligence import NexusVisualIntelligence
from src.ai_core.rag_engine import VectorStorage

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyCF3NT8VEpeqhP2gM7to6d4J7W96NIyIrU"

class VisualIntelligenceDemo:
    """Demo for NEXUS Visual Intelligence capabilities"""
    
    def __init__(self):
        """Initialize the demo"""
        # Create test directory
        self.test_dir = Path("demos/test_images")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Create vector storage for knowledge
        self.vector_storage = VectorStorage(
            storage_dir="memory/vector_db",
            backend="auto", 
            collection_name="visual_demo"
        )
        
        # Initialize NEXUS Visual Intelligence
        self.visual_intelligence = NexusVisualIntelligence(
            gemini_api_key=GEMINI_API_KEY,
            vector_storage=self.vector_storage
        )
        
        # Find or create test images
        self.test_images = {}
        self.create_test_images()
        
        # Show capabilities
        logger.info("NEXUS Visual Intelligence Demo Initialized")
        logger.info(f"Available capabilities: {self.visual_intelligence.get_capabilities()}")
    
    def create_test_images(self):
        """Create test images if they don't exist"""
        # General test image
        general_test = self.test_dir / "general_test.jpg"
        if not general_test.exists():
            self.create_general_test_image(str(general_test))
        self.test_images["general"] = str(general_test)
        
        # UI test image
        ui_test = self.test_dir / "ui_test.jpg"
        if not ui_test.exists():
            self.create_ui_test_image(str(ui_test))
        self.test_images["ui"] = str(ui_test)
        
        # Text test image
        text_test = self.test_dir / "text_test.jpg"
        if not text_test.exists():
            self.create_text_test_image(str(text_test))
        self.test_images["text"] = str(text_test)
        
        logger.info(f"Test images ready in {self.test_dir}")
    
    def create_general_test_image(self, path):
        """Create a general test image with multiple elements"""
        img = Image.new('RGB', (500, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Add a title
        draw.text((20, 20), "NEXUS Visual Test", fill=(0, 0, 0))
        
        # Add some objects
        draw.rectangle([(50, 70), (200, 150)], outline=(0, 0, 255), width=2)
        draw.text((70, 100), "Object 1", fill=(0, 0, 255))
        
        draw.ellipse([(250, 70), (350, 150)], outline=(255, 0, 0), width=2)
        draw.text((280, 100), "Object 2", fill=(255, 0, 0))
        
        # Add some text
        draw.text((50, 180), "This is a test image for NEXUS", fill=(0, 0, 0))
        draw.text((50, 210), "Visual Intelligence System", fill=(0, 0, 0))
        
        # Add a simple icon
        draw.rectangle([(400, 200), (450, 250)], fill=(0, 200, 0))
        draw.line([(400, 200), (450, 250)], fill=(255, 255, 255), width=2)
        draw.line([(450, 200), (400, 250)], fill=(255, 255, 255), width=2)
        
        img.save(path)
        logger.info(f"Created general test image: {path}")
    
    def create_ui_test_image(self, path):
        """Create a test image with UI elements"""
        img = Image.new('RGB', (600, 400), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Add header bar
        draw.rectangle([(0, 0), (600, 50)], fill=(0, 120, 215))
        draw.text((20, 15), "NEXUS Dashboard", fill=(255, 255, 255))
        
        # Add navigation menu
        draw.rectangle([(0, 50), (150, 400)], fill=(230, 230, 230))
        menu_items = ["Home", "Search", "Analytics", "Settings", "Help"]
        for i, item in enumerate(menu_items):
            y_pos = 80 + i * 40
            draw.text((20, y_pos), item, fill=(0, 0, 0))
        
        # Add content area with form
        draw.rectangle([(200, 100), (550, 350)], fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        draw.text((220, 110), "User Information", fill=(0, 0, 0))
        
        # Form fields
        form_labels = ["Name:", "Email:", "Location:"]
        for i, label in enumerate(form_labels):
            y_pos = 150 + i * 50
            draw.text((220, y_pos), label, fill=(0, 0, 0))
            draw.rectangle([(300, y_pos - 5), (500, y_pos + 25)], fill=(255, 255, 255), outline=(180, 180, 180), width=1)
        
        # Add buttons
        draw.rectangle([(300, 300), (380, 330)], fill=(0, 120, 215), outline=(0, 80, 185), width=1)
        draw.text((320, 307), "Submit", fill=(255, 255, 255))
        
        draw.rectangle([(400, 300), (480, 330)], fill=(240, 240, 240), outline=(180, 180, 180), width=1)
        draw.text((420, 307), "Cancel", fill=(0, 0, 0))
        
        img.save(path)
        logger.info(f"Created UI test image: {path}")
    
    def create_text_test_image(self, path):
        """Create a test image with primarily text content"""
        img = Image.new('RGB', (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add title
        draw.text((50, 50), "NEXUS: Adaptive AI System", fill=(0, 0, 0))
        
        # Add paragraphs
        paragraphs = [
            "NEXUS is an advanced artificial intelligence system designed to learn and adapt continuously.",
            "",
            "Key Features:",
            "• Dynamic tool orchestration and integration",
            "• Multimodal understanding with vision and text",
            "• Autonomous learning without fixed rules",
            "• Self-adaptation to available resources",
            "• Memory systems for knowledge persistence",
            "",
            "The system architecture follows these principles:",
            "1. Tools should register with a central registry",
            "2. AI should discover and combine tools dynamically",
            "3. Learning happens through continuous feedback",
            "4. Knowledge is stored in vector databases",
            "5. Adaptation is preferred over fixed behaviors",
            "",
            "This document outlines the vision for NEXUS development and expansion.",
            "The ultimate goal is to create an AI system that grows more capable over time",
            "without requiring constant reprogramming or hard-coded rules."
        ]
        
        y_position = 100
        for paragraph in paragraphs:
            draw.text((50, y_position), paragraph, fill=(0, 0, 0))
            y_position += 30
        
        img.save(path)
        logger.info(f"Created text test image: {path}")
    
    async def run_demo(self):
        """Run the Visual Intelligence demo"""
        logger.info("\n=== NEXUS VISUAL INTELLIGENCE DEMO ===")
        logger.info("This demo shows how NEXUS learns and adapts using visual intelligence")
        
        # 1. Analyze general image
        logger.info("\n=== ANALYZING GENERAL IMAGE ===")
        general_result = await self.visual_intelligence.analyze_image(
            image_path=self.test_images["general"]
        )
        
        if general_result.get("success", False):
            logger.info(f"Analysis successful using: {general_result.get('sources', [])}")
            
            if "description" in general_result:
                logger.info(f"Description: {general_result['description'][:300]}...")
            
            if "labels" in general_result and general_result["labels"]:
                labels = ", ".join([label["description"] for label in general_result["labels"][:5]])
                logger.info(f"Labels: {labels}")
            
            if "text" in general_result and isinstance(general_result["text"], dict):
                if general_result["text"].get("full_text"):
                    logger.info(f"Text: {general_result['text']['full_text']}")
        else:
            logger.error(f"General image analysis failed: {general_result.get('error')}")
        
        # 2. Analyze UI image
        logger.info("\n=== ANALYZING UI SCREENSHOT ===")
        ui_result = await self.visual_intelligence.analyze_ui(
            screenshot_path=self.test_images["ui"]
        )
        
        if ui_result.get("success", False):
            logger.info(f"UI analysis successful using: {ui_result.get('sources', [])}")
            
            if "ui_description" in ui_result:
                logger.info(f"UI Description: {ui_result['ui_description'][:300]}...")
            
            if "ui_elements" in ui_result:
                logger.info(f"Detected {len(ui_result['ui_elements'])} UI elements")
                for i, element in enumerate(ui_result.get("ui_elements", [])[:3]):
                    logger.info(f"  Element {i+1}: {element.get('type')} - '{element.get('text')}'")
        else:
            logger.error(f"UI analysis failed: {ui_result.get('error')}")
        
        # 3. Extract text from document
        logger.info("\n=== EXTRACTING TEXT FROM DOCUMENT ===")
        text_result = await self.visual_intelligence.extract_text(
            image_path=self.test_images["text"]
        )
        
        if text_result.get("success", False):
            logger.info(f"Text extraction successful using: {text_result.get('sources', [])}")
            
            if isinstance(text_result.get("text"), dict) and text_result["text"].get("full_text"):
                logger.info(f"Extracted text sample: {text_result['text']['full_text'][:300]}...")
            elif isinstance(text_result.get("text"), str):
                logger.info(f"Extracted text sample: {text_result['text'][:300]}...")
        else:
            logger.error(f"Text extraction failed: {text_result.get('error')}")
        
        # 4. Demonstrate learning by searching for similar content
        logger.info("\n=== DEMONSTRATING LEARNING AND RETRIEVAL ===")
        await asyncio.sleep(1)  # Wait for knowledge to be stored
        
        search_result = await self.visual_intelligence.find_similar_images(
            query="NEXUS dashboard with UI elements and buttons"
        )
        
        if search_result.get("success", False) and search_result.get("results"):
            logger.info(f"Found {len(search_result['results'])} related items in knowledge base")
            for i, item in enumerate(search_result.get("results", [])[:2]):
                logger.info(f"  Result {i+1}: {item.get('text', '')[:100]}...")
        else:
            logger.info("Knowledge retrieval not yet available or no results found")
        
        # Show learning progress
        stats = self.visual_intelligence.get_learning_stats()
        logger.info("\n=== LEARNING STATISTICS ===")
        logger.info(f"Images analyzed: {stats['analyzed_images']}")
        logger.info(f"Knowledge items stored: {stats['knowledge_items']}")
        
        logger.info("\n=== DEMO COMPLETE ===")
        logger.info("NEXUS Visual Intelligence has demonstrated its ability to:")
        logger.info("1. Analyze images using the best available capabilities")
        logger.info("2. Extract structured information from visual content")
        logger.info("3. Understand UI elements and screen layouts")
        logger.info("4. Store and retrieve visual knowledge")
        logger.info("5. Adapt to use whatever is available (Gemini, Cloud Vision, local vision)")
        logger.info("\nThis adaptive approach allows NEXUS to continuously learn and evolve.")

async def main():
    """Run the Visual Intelligence Demo"""
    demo = VisualIntelligenceDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
