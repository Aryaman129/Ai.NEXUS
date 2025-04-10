"""
Vision AI for NEXUS
Specialized AI for image analysis, OCR, and UI element detection
"""
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import NEXUS modules
from .shared_memory import SharedMemory

logger = logging.getLogger(__name__)

class VisionAI:
    """
    Vision AI specialist that handles all image analysis, 
    text recognition, and UI element detection tasks.
    """
    
    def __init__(self, ollama_client, memory: SharedMemory, vision_module, tool_registry):
        """Initialize the Vision AI specialist"""
        self.ollama = ollama_client
        self.memory = memory
        self.vision = vision_module  # Enhanced vision module
        self.tool_registry = tool_registry
        self.name = "vision"
        self.reasoning_log = []
        
        # Initialize advanced ML models if available
        self.yolo_detector = None
        self._initialize_ml_models()
        
        # Load vision-related tools from registry
        self.vision_tools = {}
        for tool_name, tool_info in self.tool_registry.get_all_tools().items():
            categories = tool_info.get("categories", [])
            vision_categories = ["ocr", "text_recognition", "image_analysis", 
                               "object_detection", "ui_analysis"]
            
            if any(cat in vision_categories for cat in categories):
                self.vision_tools[tool_name] = tool_info
        
        logger.info(f"NEXUS Vision AI initialized with {len(self.vision_tools)} vision tools")
    
    def _initialize_ml_models(self):
        """Initialize advanced ML models for vision if available"""
        try:
            from .ml_models import YOLODetector
            
            # Initialize YOLO detector with a small model
            self.yolo_detector = YOLODetector(model_name="yolov8n.pt")
            logger.info("YOLO detector initialized successfully")
            
        except ImportError:
            logger.info("YOLO detector not available (ultralytics not installed)")
    
    async def execute_subtask(self, subtask: Dict[str, Any], 
                            dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a vision-related subtask
        
        Args:
            subtask: The subtask definition from the Coordinator
            dependency_results: Results from dependent subtasks
            
        Returns:
            Subtask execution results
        """
        subtask_id = subtask.get("id", "unknown")
        goal = subtask.get("goal", "")
        
        logger.info(f"Vision AI executing subtask {subtask_id}: {goal}")
        
        try:
            # Log receipt of this subtask
            self.memory.log_specialist_communication(
                from_ai="coordinator",
                to_ai=self.name,
                message={"action": "received_subtask", "subtask_id": subtask_id}
            )
            
            # Determine what needs to be done
            analysis_plan = await self._create_analysis_plan(subtask, dependency_results)
            
            # Take a screenshot for analysis
            screenshot = await self.vision.take_screenshot()
            if not screenshot:
                return {
                    "status": "error",
                    "message": "Failed to take screenshot",
                    "subtask_id": subtask_id
                }
                
            # Execute the analysis plan
            results = await self._execute_analysis_plan(analysis_plan, screenshot)
            
            # Synthesize findings from all analyses
            findings = await self._synthesize_findings(results, subtask)
            
            # Return a structured result
            success = findings.get("success", False)
            final_result = {
                "status": "completed",
                "success": success,
                "findings": findings,
                "subtask_id": subtask_id,
                "analyzed_elements": findings.get("elements", []),
                "message": f"Vision analysis {'successful' if success else 'failed'} for subtask {subtask_id}"
            }
            
            # Log completion
            self.memory.log_specialist_communication(
                from_ai=self.name,
                to_ai="coordinator",
                message={"action": "completed_subtask", "subtask_id": subtask_id, "success": success}
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in Vision AI executing subtask {subtask_id}: {e}")
            
            # Log error
            self.memory.log_specialist_communication(
                from_ai=self.name,
                to_ai="coordinator",
                message={"action": "error_in_subtask", "subtask_id": subtask_id, "error": str(e)}
            )
            
            return {
                "status": "error",
                "success": False,
                "message": f"Error: {str(e)}",
                "subtask_id": subtask_id
            }
    
    async def _create_analysis_plan(self, subtask: Dict[str, Any], 
                                 dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a plan for image analysis based on the subtask"""
        goal = subtask.get("goal", "")
        
        # Create a prompt for the AI to determine the best analysis approach
        prompt = f"""
        I need to create a vision analysis plan for this task:
        
        GOAL: {goal}
        
        I have the following vision tools available:
        {', '.join(self.vision_tools.keys())}
        
        For this task, determine:
        1. What specific visual elements I need to identify
        2. Which analysis methods to use (OCR, object detection, UI element detection, etc.)
        3. The sequence of analyses to perform
        4. What specific information to extract from the image
        
        Return your analysis plan as JSON with these fields:
        - target_elements: array of elements to look for (e.g., ["buttons", "text fields", "search bar"])
        - analysis_methods: array of methods to use (e.g., ["ocr", "template_matching", "ui_detection"])
        - sequence: array of tool names to execute in order
        - information_to_extract: what specific data to extract from each element
        """
        
        # Log this reasoning step
        await self._log_reasoning("analysis_planning", prompt)
        
        # Get AI response for analysis planning
        plan_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Vision AI specialist that plans image analysis."
        )
        
        # Extract JSON from response
        plan = self._extract_json(plan_result)
        if not plan:
            logger.warning("Failed to extract analysis plan from AI response, using fallback plan")
            # Provide a fallback simple plan
            plan = {
                "target_elements": ["text", "buttons", "search bar"],
                "analysis_methods": ["ocr", "ui_detection"],
                "sequence": ["tesseract_ocr", "opencv_analysis"],
                "information_to_extract": "All visible text and UI elements"
            }
        
        logger.info(f"Created vision analysis plan with {len(plan.get('sequence', []))} steps")
        return plan
    
    async def _execute_analysis_plan(self, plan: Dict[str, Any], 
                                  screenshot) -> Dict[str, Any]:
        """Execute a vision analysis plan on a screenshot"""
        sequence = plan.get("sequence", [])
        results = {}
        
        # If no sequences specified, use default OCR and UI detection
        if not sequence:
            sequence = ["tesseract_ocr", "opencv_analysis"]
        
        # Add YOLO detection if available and not already in sequence
        if self.yolo_detector is not None and "yolo_detection" not in sequence:
            sequence.append("yolo_detection")
        
        # Execute each tool in the sequence
        for tool_name in sequence:
            if tool_name == "yolo_detection" and self.yolo_detector is not None:
                # Use YOLO detector directly
                try:
                    logger.info("Executing YOLO detection")
                    
                    # Detect objects and UI elements
                    object_results = await self.yolo_detector.detect_objects(screenshot)
                    ui_results = await self.yolo_detector.detect_ui_elements(screenshot)
                    
                    # Store the results
                    results["yolo_detection"] = {
                        "objects": object_results.get("detections", []),
                        "ui_elements": ui_results.get("ui_elements", []),
                        "inference_time": object_results.get("inference_time", 0)
                    }
                    
                    # Update tool statistics in memory
                    self.memory.update_tool_stats(
                        "yolo_detection", 
                        success=object_results.get("success", False)
                    )
                    
                except Exception as e:
                    logger.error(f"Error executing YOLO detection: {e}")
                    results["yolo_detection"] = {"error": str(e)}
                
            elif tool_name not in self.vision_tools:
                logger.warning(f"Tool {tool_name} not found in vision tools, skipping")
                continue
                
            else:
                try:
                    # Convert PIL Image to a file for tools that expect a file path
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        screenshot_path = tmp.name
                        screenshot.save(screenshot_path)
                    
                    # Execute the tool
                    logger.info(f"Executing vision tool: {tool_name}")
                    tool_result = await self.tool_registry.execute_tool(
                        tool_name, 
                        screenshot_path=screenshot_path
                    )
                    
                    # Store the result
                    results[tool_name] = tool_result
                    
                    # Update tool statistics in memory
                    self.memory.update_tool_stats(
                        tool_name, 
                        success="error" not in tool_result
                    )
                    
                except Exception as e:
                    logger.error(f"Error executing vision tool {tool_name}: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _synthesize_findings(self, tool_results: Dict[str, Any], 
                               subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings from all analysis tools into a coherent result"""
        goal = subtask.get("goal", "")
        
        # Create a prompt for the AI to synthesize findings
        prompt = f"""
        I need to synthesize findings from multiple vision analysis tools for this task:
        
        GOAL: {goal}
        
        Tool results:
        {json.dumps(tool_results, indent=2)}
        
        Please analyze these results and:
        1. Identify all detected UI elements, text, and objects
        2. Determine if we found what we were looking for
        3. Extract the most relevant information for this task
        4. Provide confidence scores for our findings
        
        Return your synthesis as JSON with these fields:
        - success: boolean indicating if we found what we needed
        - elements: array of detected elements with positions and types
        - extracted_text: all text found that's relevant to the task
        - confidence: overall confidence score (0-1)
        - notes: any important observations about the findings
        """
        
        # Log this reasoning step
        await self._log_reasoning("findings_synthesis", prompt)
        
        # Get AI response for synthesis
        synthesis_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Vision AI specialist that synthesizes findings."
        )
        
        # Extract JSON from response
        findings = self._extract_json(synthesis_result)
        if not findings:
            logger.warning("Failed to extract findings from AI response, using basic findings")
            
            # Create basic findings from the raw results
            all_elements = []
            all_text = []
            
            # Extract elements and text from each tool result
            for tool_name, result in tool_results.items():
                if "elements" in result:
                    all_elements.extend(result["elements"])
                if "text" in result:
                    all_text.append(result["text"])
            
            findings = {
                "success": len(all_elements) > 0 or len(all_text) > 0,
                "elements": all_elements,
                "extracted_text": " ".join(all_text),
                "confidence": 0.5,  # Medium confidence as this is a fallback
                "notes": ["Synthesized from raw tool results due to synthesis failure"]
            }
        
        return findings
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from AI response text"""
        try:
            # Find JSON-like content between curly braces
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return {}
    
    async def _log_reasoning(self, context: str, prompt: str, response: str = None) -> None:
        """Log reasoning steps for debugging and improvement"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "prompt": prompt,
            "response": response[:1000] + ("..." if response and len(response) > 1000 else "") if response else "Pending"
        }
        
        self.reasoning_log.append(log_entry)
        return log_entry
