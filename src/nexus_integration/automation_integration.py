"""
NEXUS Adaptive Automation Integration

This module integrates the adaptive automation capabilities into the 
broader NEXUS architecture, connecting it with the central registry,
memory systems, and other NEXUS components.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import NEXUS core components
from src.ai_core.automation.mouse_controller import MouseController
from src.ai_core.automation.keyboard_controller import KeyboardController
from src.ai_core.automation.safety_manager import SafetyManager
from src.ai_core.automation.clarification_engine import ClarificationEngine
from src.ai_core.screen_analysis.visual_memory import VisualMemorySystem

# Import NEXUS tool registry system
# (This would be your existing registry system - placeholder imports for now)
from src.nexus_core.registry import ToolRegistry, Tool, ToolCategory
from src.nexus_core.memory_system import GlobalMemorySystem
from src.nexus_core.orchestrator import AIOrchestrator

class AdaptiveAutomationManager:
    """
    Integrates the adaptive automation capabilities with the NEXUS core system.
    
    This manager:
    1. Registers automation tools with the central NEXUS registry
    2. Connects the visual memory system with the global memory system
    3. Provides performance metrics for the AI orchestrator to optimize tool selection
    4. Handles clarification through the appropriate NEXUS channels
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the automation manager with configuration"""
        self.config = config or {}
        
        # Initialize component paths
        memory_base = self.config.get("memory_base_path", "memory/automation")
        os.makedirs(memory_base, exist_ok=True)
        
        # Create core automation components
        self.safety_manager = SafetyManager(
            config=self.config.get("safety_config", {})
        )
        
        self.mouse_controller = MouseController(
            safety_manager=self.safety_manager,
            config=self.config.get("mouse_config", {})
        )
        
        self.keyboard_controller = KeyboardController(
            safety_manager=self.safety_manager,
            config=self.config.get("keyboard_config", {})
        )
        
        # Create visual memory system
        visual_memory_path = os.path.join(memory_base, "visual_memory")
        os.makedirs(visual_memory_path, exist_ok=True)
        
        self.visual_memory = VisualMemorySystem(config={
            "memory_path": visual_memory_path,
            "max_patterns": self.config.get("max_visual_patterns", 10000),
            "similarity_threshold": self.config.get("visual_similarity_threshold", 0.7),
            "enable_learning": True
        })
        
        # Create clarification engine
        clarification_path = os.path.join(memory_base, "clarification_memory")
        os.makedirs(clarification_path, exist_ok=True)
        
        self.clarification_engine = ClarificationEngine(config={
            "memory_path": clarification_path,
            "confidence_threshold": self.config.get("clarification_threshold", 0.7),
            "enable_learning": True
        })
        
        # Reference to the NEXUS registry
        self._registry = None
        self._orchestrator = None
        self._global_memory = None
        
        # Register clarification callback - connecting to NEXUS communication channels
        self.clarification_engine.set_response_callback(self._nexus_clarification_callback)
        
        logger.info("AdaptiveAutomationManager initialized")
    
    def connect_to_nexus(self, registry: 'ToolRegistry', 
                         orchestrator: 'AIOrchestrator',
                         global_memory: 'GlobalMemorySystem'):
        """Connect automation manager to NEXUS core systems"""
        self._registry = registry
        self._orchestrator = orchestrator
        self._global_memory = global_memory
        
        # Register with the memory system to sync visual memories
        if self._global_memory:
            self._connect_memory_systems()
        
        # Register tools with the registry
        if self._registry:
            self._register_automation_tools()
        
        logger.info("Connected adaptive automation to NEXUS core systems")
    
    def _connect_memory_systems(self):
        """Connect visual memory with the global memory system"""
        # Register callbacks to sync memories both ways
        self.visual_memory.register_sync_callback(self._sync_to_global_memory)
        
        # Register a callback for the global memory to update visual memory
        self._global_memory.register_memory_callback(
            "visual_patterns", self._sync_from_global_memory
        )
        
        # Perform initial sync
        self._sync_visual_memories()
        
        logger.info("Connected visual memory system to global memory")
    
    def _sync_visual_memories(self):
        """Perform two-way sync between visual and global memories"""
        # Get all patterns from visual memory
        patterns = self.visual_memory.get_all_patterns()
        
        # Upload to global memory
        for pattern_id, pattern in patterns.items():
            self._global_memory.store_memory(
                memory_type="visual_patterns",
                memory_id=f"visual_{pattern_id}",
                content=pattern,
                metadata={
                    "source": "visual_memory",
                    "type": pattern.get("type", "unknown"),
                    "success_rate": pattern.get("success_rate", 0),
                    "confidence": pattern.get("confidence", 0)
                }
            )
        
        # Get relevant memories from global memory
        global_memories = self._global_memory.query_memories(
            memory_type="ui_elements",
            limit=1000
        )
        
        # Import into visual memory
        for memory in global_memories:
            if "visual_signature" in memory.content:
                self.visual_memory.store_pattern(memory.content)
        
        logger.info(f"Synced {len(patterns)} visual patterns with global memory")
    
    def _sync_to_global_memory(self, pattern_id: str, pattern: Dict):
        """Callback when visual memory is updated to sync to global memory"""
        if self._global_memory:
            self._global_memory.store_memory(
                memory_type="visual_patterns",
                memory_id=f"visual_{pattern_id}",
                content=pattern,
                metadata={
                    "source": "visual_memory",
                    "type": pattern.get("type", "unknown"),
                    "success_rate": pattern.get("success_rate", 0),
                    "confidence": pattern.get("confidence", 0)
                }
            )
    
    def _sync_from_global_memory(self, memory_update: Dict):
        """Callback when global memory has a relevant update"""
        if "content" in memory_update and "visual_signature" in memory_update["content"]:
            self.visual_memory.store_pattern(memory_update["content"])
    
    def _register_automation_tools(self):
        """Register automation tools with the NEXUS central registry"""
        if not self._registry:
            logger.warning("No registry available to register automation tools")
            return
        
        # Register mouse control tools
        self._registry.register_tool(
            Tool(
                name="move_mouse_to",
                description="Move mouse cursor to specific coordinates with adaptive speed",
                category=ToolCategory.AUTOMATION,
                function=self.move_mouse_to,
                parameters=[
                    {"name": "x", "type": "integer", "description": "X coordinate"},
                    {"name": "y", "type": "integer", "description": "Y coordinate"},
                    {"name": "speed_factor", "type": "float", "description": "Speed multiplier", "default": 1.0},
                    {"name": "safety_override", "type": "boolean", "description": "Override safety checks", "default": False}
                ],
                success_metrics={
                    "avg_movement_time": self.mouse_controller.get_performance_stats().get("avg_movement_time", 0),
                    "success_rate": 0.95  # Initial estimate
                }
            )
        )
        
        self._registry.register_tool(
            Tool(
                name="click_mouse",
                description="Click the mouse at current or specified position",
                category=ToolCategory.AUTOMATION,
                function=self.click_mouse,
                parameters=[
                    {"name": "x", "type": "integer", "description": "X coordinate", "optional": True},
                    {"name": "y", "type": "integer", "description": "Y coordinate", "optional": True},
                    {"name": "button", "type": "string", "description": "Mouse button (left, right, middle)", "default": "left"},
                    {"name": "clicks", "type": "integer", "description": "Number of clicks", "default": 1},
                    {"name": "safety_override", "type": "boolean", "description": "Override safety checks", "default": False}
                ],
                success_metrics={
                    "success_rate": 0.9  # Initial estimate
                }
            )
        )
        
        # Register keyboard control tools
        self._registry.register_tool(
            Tool(
                name="type_text",
                description="Type text with adaptive speed based on performance",
                category=ToolCategory.AUTOMATION,
                function=self.type_text,
                parameters=[
                    {"name": "text", "type": "string", "description": "Text to type"},
                    {"name": "delay", "type": "float", "description": "Delay between keystrokes", "optional": True}
                ],
                success_metrics={
                    "avg_typing_speed": self.keyboard_controller.get_performance_stats().get("avg_typing_speed", 0),
                    "success_rate": 0.95  # Initial estimate
                }
            )
        )
        
        self._registry.register_tool(
            Tool(
                name="press_keys",
                description="Press key combination (e.g. for shortcuts)",
                category=ToolCategory.AUTOMATION,
                function=self.press_keys,
                parameters=[
                    {"name": "keys", "type": "array", "description": "List of keys to press simultaneously"}
                ],
                success_metrics={
                    "success_rate": 0.9  # Initial estimate
                }
            )
        )
        
        # Register visual detection enhancement tool
        self._registry.register_tool(
            Tool(
                name="enhance_ui_detection",
                description="Enhance UI element detection using visual memory",
                category=ToolCategory.SCREEN_ANALYSIS,
                function=self.enhance_ui_detection,
                parameters=[
                    {"name": "ui_elements", "type": "array", "description": "List of detected UI elements"},
                    {"name": "screen_image", "type": "object", "description": "Screen image as numpy array"},
                    {"name": "context", "type": "object", "description": "Context information like window title", "optional": True}
                ],
                success_metrics={
                    "enhancement_rate": 0.8,  # Initial estimate
                    "false_positive_rate": 0.05  # Initial estimate
                }
            )
        )
        
        logger.info("Registered automation tools with NEXUS registry")
    
    def _nexus_clarification_callback(self, question: str) -> str:
        """Forward clarification questions to the NEXUS orchestrator"""
        if self._orchestrator and hasattr(self._orchestrator, "ask_user"):
            # Let the orchestrator handle user interaction
            return self._orchestrator.ask_user(
                question=question,
                context="automation_clarification",
                options=["yes", "no", "more_info"]
            )
        else:
            # Fallback - direct console input
            logger.warning("No orchestrator for clarification, using console fallback")
            print(f"\nNEXUS needs clarification: {question}")
            return input("Your response: ")
    
    # Tool implementations that connect to the underlying controllers
    
    def move_mouse_to(self, x: int, y: int, speed_factor: float = 1.0, 
                     safety_override: bool = False) -> Dict:
        """Move mouse to specified coordinates with adaptive learning"""
        # Check confidence based on past performance
        confidence = self._get_confidence_for_action("move_mouse", {"x": x, "y": y})
        
        # Ask for clarification if confidence is low and safety_override is False
        if not safety_override and confidence < 0.7:
            result = self.clarification_engine.ask_for_clarification(
                scenario="mouse_movement",
                context={"x": x, "y": y},
                confidence=confidence
            )
            
            if not result["proceed"]:
                return {"success": False, "reason": "user_declined"}
            
            confidence = result["confidence"]
        
        # Execute the action with confidence-adjusted speed
        adjusted_speed = speed_factor * (0.5 + (confidence * 0.5))  # 0.5-1.0 × speed_factor
        result = self.mouse_controller.move_to(x, y, adjusted_speed, safety_override=safety_override)
        
        # Update success metrics for the registry
        if self._registry:
            self._update_tool_metrics("move_mouse_to", result)
        
        return result
    
    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, 
                   button: str = "left", clicks: int = 1, 
                   safety_override: bool = False) -> Dict:
        """Click mouse with adaptive learning"""
        # Get current position if not specified
        if x is None or y is None:
            pos = self.mouse_controller.get_position()
            x = x if x is not None else pos["x"]
            y = y if y is not None else pos["y"]
        
        # Check confidence based on past performance
        confidence = self._get_confidence_for_action("click_mouse", 
                                                 {"x": x, "y": y, "button": button})
        
        # Ask for clarification if confidence is low and safety_override is False
        if not safety_override and confidence < 0.7:
            result = self.clarification_engine.ask_for_clarification(
                scenario="mouse_click",
                context={"x": x, "y": y, "button": button},
                confidence=confidence
            )
            
            if not result["proceed"]:
                return {"success": False, "reason": "user_declined"}
            
            confidence = result["confidence"]
        
        # Execute the action
        result = self.mouse_controller.click(x, y, button, clicks, safety_override)
        
        # Update success metrics for the registry
        if self._registry:
            self._update_tool_metrics("click_mouse", result)
        
        return result
    
    def type_text(self, text: str, delay: Optional[float] = None) -> Dict:
        """Type text with adaptive learning for speed and accuracy"""
        # Check confidence based on past performance and text content
        confidence = self._get_confidence_for_action("type_text", {"text": text[:50]})
        
        # Check if text contains dangerous patterns (e.g., system commands)
        if self.safety_manager.is_dangerous_text(text):
            confidence *= 0.5  # Reduce confidence for potentially dangerous text
        
        # Ask for clarification if confidence is low
        if confidence < 0.7:
            result = self.clarification_engine.ask_for_clarification(
                scenario="keyboard_input",
                context={"text": text[:50] + ("..." if len(text) > 50 else "")},
                confidence=confidence
            )
            
            if not result["proceed"]:
                return {"success": False, "reason": "user_declined"}
            
            confidence = result["confidence"]
        
        # Adjust typing speed based on confidence
        if delay is None:
            # Higher confidence = faster typing
            typing_speed = self.keyboard_controller.get_default_typing_speed() * (0.5 + (confidence * 0.5))
            self.keyboard_controller.set_typing_speed(typing_speed)
        
        # Execute the action
        result = self.keyboard_controller.type_text(text, delay)
        
        # Update success metrics for the registry
        if self._registry:
            self._update_tool_metrics("type_text", result)
        
        return result
    
    def press_keys(self, keys: List[str]) -> Dict:
        """Press keys with adaptive safety checks"""
        # Check if key combination is potentially dangerous
        is_dangerous = self.safety_manager.is_dangerous_key_combination(keys)
        
        # Check confidence based on past performance and keys
        confidence = self._get_confidence_for_action("press_keys", {"keys": keys})
        
        if is_dangerous:
            confidence *= 0.3  # Significantly reduce confidence for dangerous combinations
        
        # Ask for clarification if confidence is low or combination is dangerous
        if confidence < 0.8 or is_dangerous:
            result = self.clarification_engine.ask_for_clarification(
                scenario="keyboard_shortcut",
                context={"keys": keys, "dangerous": is_dangerous},
                confidence=confidence
            )
            
            if not result["proceed"]:
                return {"success": False, "reason": "user_declined"}
            
            confidence = result["confidence"]
        
        # Execute the action
        if len(keys) == 1:
            result = self.keyboard_controller.press_key(keys[0])
        else:
            result = self.keyboard_controller.press_keys(keys)
        
        # Update success metrics for the registry
        if self._registry:
            self._update_tool_metrics("press_keys", result)
        
        return result
    
    def enhance_ui_detection(self, ui_elements: List[Dict], screen_image: Any, 
                           context: Optional[Dict] = None) -> List[Dict]:
        """Enhance UI detection using visual memory system"""
        # The context may include window title, application name, etc.
        layout_context = context or {}
        
        # Call visual memory to enhance detection
        enhanced_elements = self.visual_memory.enhance_detection(
            layout_results=layout_context,
            ui_elements=ui_elements,
            screen_image=screen_image
        )
        
        # Record this detection in the global memory for future learning
        if self._global_memory:
            # Store the detection context
            self._global_memory.store_memory(
                memory_type="ui_detection",
                content={
                    "timestamp": "auto",
                    "context": layout_context,
                    "element_count": len(ui_elements),
                    "enhanced_count": len(enhanced_elements)
                },
                metadata={
                    "application": layout_context.get("application", "unknown"),
                    "window_title": layout_context.get("window_title", "unknown")
                }
            )
        
        # Update success metrics
        enhanced_count = sum(1 for e in enhanced_elements if "enhanced_confidence" in e)
        if self._registry and ui_elements:
            self._registry.update_tool_metrics(
                "enhance_ui_detection",
                {
                    "enhancement_rate": enhanced_count / len(ui_elements),
                    "success": True
                }
            )
        
        return enhanced_elements
    
    def _get_confidence_for_action(self, action_type: str, context: Dict) -> float:
        """Calculate confidence for an action based on past performance and context"""
        # Start with a base confidence
        base_confidence = 0.7
        
        # Adjust based on the tool's success metrics from the registry
        if self._registry:
            tool_metrics = self._registry.get_tool_metrics(action_type)
            if tool_metrics and "success_rate" in tool_metrics:
                base_confidence = tool_metrics["success_rate"]
        
        # Get visual context to adjust confidence
        if action_type in ["click_mouse", "move_mouse"] and "x" in context and "y" in context:
            # Check if we are operating in a familiar screen region
            familiar_regions = self.visual_memory.find_patterns_at_location(
                x=context["x"], 
                y=context["y"],
                radius=20
            )
            
            if familiar_regions:
                # Average success rate of patterns in this region
                avg_success = sum(p.get("success_rate", 0.5) for p in familiar_regions) / len(familiar_regions)
                # Boost confidence for familiar regions with high success
                base_confidence = base_confidence * 0.7 + avg_success * 0.3
        
        # Check with clarification engine for historical confidence on similar actions
        similar_scenarios = self.clarification_engine.find_similar_scenarios(
            action_type, context
        )
        
        if similar_scenarios:
            # Average confidence from similar scenarios
            scenario_confidence = sum(s["final_confidence"] for s in similar_scenarios) / len(similar_scenarios)
            # Blend with base confidence
            base_confidence = base_confidence * 0.6 + scenario_confidence * 0.4
        
        # Ensure confidence is between 0.1 and 1.0
        return max(0.1, min(1.0, base_confidence))
    
    def _update_tool_metrics(self, tool_name: str, result: Dict):
        """Update success metrics for a tool in the registry"""
        if not self._registry:
            return
            
        metrics_update = {
            "success": result.get("success", False)
        }
        
        # Add tool-specific metrics
        if tool_name == "move_mouse_to":
            mouse_stats = self.mouse_controller.get_performance_stats()
            metrics_update.update({
                "avg_movement_time": mouse_stats.get("avg_movement_time", 0),
                "movement_speed": mouse_stats.get("movement_speed", 0)
            })
        elif tool_name == "type_text":
            keyboard_stats = self.keyboard_controller.get_performance_stats()
            metrics_update.update({
                "avg_typing_speed": keyboard_stats.get("avg_typing_speed", 0)
            })
        
        # Update the registry
        self._registry.update_tool_metrics(tool_name, metrics_update)

# Main function to demonstrate integration
def integrate_with_nexus(config_path: Optional[str] = None) -> AdaptiveAutomationManager:
    """
    Integrate the adaptive automation system with the NEXUS core.
    
    This is the main function to call when setting up NEXUS with automation.
    """
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Initialize the automation manager
    automation_manager = AdaptiveAutomationManager(config)
    
    # Import NEXUS core components (these would be your actual imports)
    # For this example, we create placeholders
    try:
        from src.nexus_core.registry import ToolRegistry
        from src.nexus_core.memory_system import GlobalMemorySystem
        from src.nexus_core.orchestrator import AIOrchestrator
        
        # Get or create NEXUS components
        registry = ToolRegistry.get_instance()
        memory_system = GlobalMemorySystem.get_instance()
        orchestrator = AIOrchestrator.get_instance()
        
        # Connect the automation manager
        automation_manager.connect_to_nexus(
            registry=registry,
            orchestrator=orchestrator,
            global_memory=memory_system
        )
        
        logger.info("Successfully integrated automation with NEXUS core")
        
    except ImportError:
        logger.warning("NEXUS core components not found. Running in standalone mode.")
        # When running without NEXUS core, we still return the manager for direct use
    
    return automation_manager

if __name__ == "__main__":
    # This file can be run directly for testing
    logging.basicConfig(level=logging.INFO)
    print("Initializing NEXUS Adaptive Automation Integration...")
    
    automation_manager = integrate_with_nexus()
    
    print("\nAdaptive Automation System is ready.")
    print("In a full NEXUS deployment, this module would connect to:")
    print("1. The central tool registry for dynamic tool discovery")
    print("2. The global memory system for sharing UI patterns")
    print("3. The AI orchestrator for clarification management")
    
    # Show some debug info
    mouse_stats = automation_manager.mouse_controller.get_performance_stats()
    keyboard_stats = automation_manager.keyboard_controller.get_performance_stats()
    
    print("\nCurrent Performance Metrics:")
    print(f"Mouse movement time: {mouse_stats.get('avg_movement_time', 0):.4f}s")
    print(f"Keyboard typing speed: {keyboard_stats.get('avg_typing_speed', 0):.2f} chars/sec")
    
    # Example of registering a custom clarification callback
    def custom_callback(question):
        print(f"\n[NEXUS AI] {question}")
        return input("Your response: ")
    
    print("\nTo use the automation system directly:")
    print("automation_manager.clarification_engine.set_response_callback(custom_callback)")
    print("automation_manager.move_mouse_to(x=500, y=500)")
    print("automation_manager.type_text('Hello NEXUS!')")
