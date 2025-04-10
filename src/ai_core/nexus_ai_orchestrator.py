"""
NEXUS AI Orchestrator
Main orchestration layer that manages all NEXUS AI components
"""
import logging
import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional, Union

from .shared_memory import SharedMemory
from .coordinator_ai import CoordinatorAI
from .vision_ai import VisionAI
from .research_ai import ResearchAI
from .automation_ai import AutomationAI

# Import advanced components if available
try:
    from .rag_engine import KnowledgeManager, VectorStorage
    from .rag_engine.multimodal_knowledge import MultimodalKnowledgeManager
    from .rag_engine.visual_intelligence import VisualIntelligence
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from .ml_models import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class NexusAIOrchestrator:
    """
    NEXUS AI Orchestrator - Main integration layer that manages all AI components
    
    This orchestrates the interactions between specialist AIs and manages the overall
    execution flow of NEXUS.
    """
    
    def __init__(self, ollama_client, tool_registry):
        """Initialize the NEXUS AI Orchestrator"""
        self.ollama = ollama_client
        self.tool_registry = tool_registry
        
        # Initialize shared memory
        self.memory = SharedMemory()
        
        # Initialize knowledge components if available
        self.vector_storage = None
        self.knowledge_manager = None
        self.multimodal_knowledge = None
        self.visual_intelligence = None
        
        if RAG_AVAILABLE:
            try:
                # Initialize vector storage
                self.vector_storage = VectorStorage(
                    storage_dir="memory/vector_db",
                    backend="auto",
                    collection_name="nexus_knowledge"
                )
                
                # Initialize knowledge manager
                self.knowledge_manager = KnowledgeManager(
                    ollama_client=ollama_client,
                    vector_storage=self.vector_storage,
                    knowledge_base_dir="memory/knowledge"
                )
                
                # Initialize multimodal knowledge
                self.multimodal_knowledge = MultimodalKnowledgeManager(
                    ollama_client=ollama_client,
                    vector_storage=self.vector_storage,
                    knowledge_base_dir="memory/multimodal_knowledge"
                )
                
                # Initialize visual intelligence
                self.visual_intelligence = VisualIntelligence(
                    vector_storage=self.vector_storage,
                    ollama_client=ollama_client,
                    knowledge_base_dir="memory/visual_intelligence"
                )
                
                logger.info("RAG components initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing RAG components: {e}")
        
        # Initialize specialist AIs
        self.coordinator = CoordinatorAI(ollama_client, self.memory, tool_registry)
        self.vision = VisionAI(ollama_client, self.memory, None, tool_registry)
        self.research = ResearchAI(ollama_client, self.memory, tool_registry)
        self.automation = AutomationAI(ollama_client, self.memory, tool_registry)
        
        # Register specialists
        self.specialists = {
            "coordinator": self.coordinator,
            "vision": self.vision,
            "research": self.research,
            "automation": self.automation
        }
        
        # Store active task state
        self.active_task = None
        self.subtasks = {}
        self.execution_log = []
        
        logger.info("NEXUS AI Orchestrator initialized")
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a complete task using the NEXUS architecture
        
        Args:
            task_description: Description of the task to perform
            
        Returns:
            Dictionary with task results
        """
        logger.info(f"Processing task: {task_description}")
        
        # Store task in memory context
        self.memory.set_context("current_task", task_description)
        self.active_task = task_description
        
        # Reset execution state
        self.subtasks = {}
        self.execution_log = []
        
        # Enhance task with knowledge if available
        enhanced_task = task_description
        if self.knowledge_manager:
            enhanced_task = await self.knowledge_manager.enhance_with_knowledge(
                original_prompt=task_description,
                task_description=task_description
            )
            logger.info("Enhanced task with relevant knowledge")
        
        # 1. Task Decomposition
        subtasks = await self.coordinator.decompose_task(enhanced_task)
        
        # Store subtasks in memory
        self.memory.set_context("task_decomposition", subtasks)
        
        # 2. Execute Subtasks
        results = {}
        active_specialists = set()
        
        for subtask in subtasks:
            subtask_id = subtask["id"]
            specialist_name = subtask["specialist"]
            
            # Get dependencies
            dependencies = {}
            for dep_id in subtask.get("dependencies", []):
                if dep_id in results:
                    dependencies[dep_id] = results[dep_id]
            
            # Get specialist
            if specialist_name in self.specialists:
                specialist = self.specialists[specialist_name]
                active_specialists.add(specialist_name)
                
                # Log execution
                logger.info(f"Executing subtask {subtask_id} with {specialist_name}")
                execution_start = {
                    "id": subtask_id, 
                    "specialist": specialist_name,
                    "status": "started",
                    "timestamp": time.time()
                }
                self.execution_log.append(execution_start)
                
                # Execute subtask
                subtask_result = await specialist.execute_subtask(subtask, dependencies)
                
                # Store result
                results[subtask_id] = subtask_result
                
                # Log completion
                execution_complete = {
                    "id": subtask_id, 
                    "specialist": specialist_name,
                    "status": "completed",
                    "success": subtask_result.get("success", False),
                    "timestamp": time.time()
                }
                self.execution_log.append(execution_complete)
            else:
                logger.error(f"Unknown specialist: {specialist_name}")
                results[subtask_id] = {
                    "success": False,
                    "error": f"Unknown specialist: {specialist_name}"
                }
        
        # Store active specialists in memory
        self.memory.set_context("active_specialists", list(active_specialists))
        
        # 3. Synthesize Results
        final_result = await self.coordinator.synthesize_results(results, task_description)
        
        # Add execution metadata
        final_result["execution"] = {
            "subtasks": len(subtasks),
            "specialists_used": list(active_specialists),
            "timestamp": time.time()
        }
        
        # Log task completion
        logger.info(f"Task completed with success={final_result.get('success', False)}")
        
        # 4. Learn from execution if knowledge manager is available
        if self.knowledge_manager:
            try:
                await self.knowledge_manager.analyze_task_success(
                    task_description=task_description,
                    execution_log=self.execution_log,
                    success=final_result.get("success", False)
                )
            except Exception as e:
                logger.error(f"Error learning from task execution: {e}")
        
        return final_result
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image using all available visual intelligence capabilities
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        if self.visual_intelligence:
            return await self.visual_intelligence.analyze_image(image_path=image_path)
        elif self.vision:
            # Fallback to basic vision capabilities
            return await self.vision.analyze_image(image_path)
        else:
            return {
                "success": False,
                "error": "No visual analysis capabilities available"
            }
    
    async def search_and_learn(self, query: str, include_images: bool = True) -> Dict[str, Any]:
        """
        Search for information and learn from the results
        
        Args:
            query: Search query
            include_images: Whether to include images in results
            
        Returns:
            Dictionary with search results and learning outcomes
        """
        if self.multimodal_knowledge:
            return await self.multimodal_knowledge.search_and_learn(
                query=query,
                include_images=include_images
            )
        elif self.research:
            # Fallback to basic research capabilities
            return await self.research.search_web(query)
        else:
            return {
                "success": False,
                "error": "No search capabilities available"
            }
    
    async def analyze_ui(self, screenshot_path: str = None) -> Dict[str, Any]:
        """
        Analyze UI elements in a screenshot
        
        Args:
            screenshot_path: Path to screenshot file
            
        Returns:
            Dictionary with UI analysis results
        """
        if self.visual_intelligence:
            return await self.visual_intelligence.analyze_ui(screenshot_path=screenshot_path)
        elif self.vision:
            # Fallback to basic vision capabilities
            return await self.vision.analyze_screenshot(screenshot_path)
        else:
            return {
                "success": False,
                "error": "No UI analysis capabilities available"
            }
    
    def get_available_capabilities(self) -> Dict[str, Any]:
        """Get information about available capabilities in this NEXUS instance"""
        capabilities = {
            "specialists": list(self.specialists.keys()),
            "rag_available": RAG_AVAILABLE,
            "yolo_available": YOLO_AVAILABLE,
            "knowledge_components": {
                "vector_storage": self.vector_storage is not None,
                "knowledge_manager": self.knowledge_manager is not None,
                "multimodal_knowledge": self.multimodal_knowledge is not None,
                "visual_intelligence": self.visual_intelligence is not None
            }
        }
        
        # Add vision capabilities
        vision_capabilities = []
        if self.visual_intelligence and hasattr(self.visual_intelligence, "cloud_vision") and self.visual_intelligence.cloud_vision:
            vision_capabilities.append("cloud_vision")
        if YOLO_AVAILABLE:
            vision_capabilities.append("yolo")
        if self.vision:
            vision_capabilities.append("basic_vision")
        
        capabilities["vision_capabilities"] = vision_capabilities
        
        return capabilities
