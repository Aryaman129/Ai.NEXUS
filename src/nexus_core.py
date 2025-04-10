"""
NEXUS - Autonomous AI Orchestration System
Main Core Implementation
"""
import asyncio
import cv2
import json
import logging
import os
import pyautogui
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import sys
import random
from ai_services.research_service import ResearchService

# Add the AI-Automation-System directory to the Python path
sys.path.append(str(Path(__file__).parent / "AI-Automation-System"))

# Import the existing modules
from integrations.ollama_integration import OllamaIntegration
from integrations.ai_screen_agent import AIScreenAgent

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NexusAutonomousDemo:
    """NEXUS Autonomous AI demo that uses natural language to control the computer"""
    
    def __init__(self):
        """Initialize the NEXUS autonomous agent"""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize the Ollama integration
        self.ollama = OllamaIntegration()
        
        # Initialize the screen agent
        self.screen_agent = AIScreenAgent(ollama_client=self.ollama)
        self.screen_size = self.screen_agent.screen_size
        
        # Safe mode (if True, only simulates actions)
        self.safe_mode = False
        
        # Initialize memory system
        self.memory = self._load_memory()
        
        # Track unsuccessful attempts to learn from failures
        self.current_attempt = {
            "task": "",
            "attempts": 0,
            "elements_tried": []
        }
        
        # Initialize tool registry
        self.tool_registry = NexusToolRegistry()
        self.tool_registry.register_tool(
            "tesseract_ocr",
            self._execute_tesseract_ocr,
            categories=["text_recognition", "ocr"],
            description="Tesseract OCR for text recognition"
        )
        self.tool_registry.register_tool(
            "opencv_analysis",
            self._execute_opencv_analysis,
            categories=["image_analysis", "object_detection"],
            description="OpenCV for image analysis and object detection"
        )
        self.tool_registry.register_tool(
            "duckduckgo_search",
            self._execute_duckduckgo_search,
            categories=["web_search", "information_retrieval"],
            description="DuckDuckGo search for web-based information retrieval"
        )
        
        # Initialize Nexus intelligence
        self.nexus_intelligence = NexusIntelligence(self.ollama, self.tool_registry)
        
        logger.info("NEXUS Autonomous Demo initialized successfully")
    
    # ... existing code ...

    async def execute_task(self, task_plan: Dict) -> Dict:
        """Execute a structured task plan with AI orchestration"""
        logger.info(f"Executing task for application: {task_plan.get('application', 'unknown')}")
        
        # Determine if we should use AI orchestration
        use_ai_orchestration = True  # Eventually this could be a configurable setting
        
        if use_ai_orchestration:
            # Let the AI handle tool selection and orchestration
            return await self.execute_task_with_ai_orchestration(task_plan)
        else:
            # Use the traditional execution approach
            results = {
                "success": False,
                "steps_completed": 0,
                "steps_total": len(task_plan.get("steps", [])),
                "notes": []
            }
            
            # Launch the application if specified
            app_name = task_plan.get("application", "")
            if app_name:
                launched = await self.screen_agent.launch_application(app_name)
                if not launched:
                    results["notes"].append(f"Failed to launch application: {app_name}")
                    return results
                results["notes"].append(f"Launched application: {app_name}")
                
            # Analyze the screen to extract UI elements
            screenshot_path = await self.screen_agent._take_screenshot()
            screen_analysis = await self._analyze_screen(screenshot_path, task_plan)
            
            # Execute each step in the task
            steps = task_plan.get("steps", [])
            for i, step in enumerate(steps):
                step_result = await self._execute_step(step, screen_analysis)
                
                if step_result.get("success", False):
                    results["steps_completed"] += 1
                    results["notes"].append(f"Completed step {i+1}: {step.get('description', '')}")
                else:
                    results["notes"].append(f"Failed step {i+1}: {step.get('description', '')} - {step_result.get('error', 'unknown error')}")
                    break
                    
                # Take a new screenshot for the next step if needed
                if i < len(steps) - 1:
                    screenshot_path = await self.screen_agent._take_screenshot()
                    screen_analysis = await self._analyze_screen(screenshot_path, task_plan)
                
            # Determine overall success
            results["success"] = results["steps_completed"] == results["steps_total"]
            
            # Learn from the execution
            if results["success"]:
                self._learn_from_success(task_plan, screen_analysis.get("elements", []), [])
            else:
                self._learn_from_failure(task_plan, self.current_attempt, 
                                       f"Completed {results['steps_completed']}/{results['steps_total']} steps")
            
            return results

    # ... existing code ...

    async def execute_task_with_ai_orchestration(self, task_plan: Dict) -> Dict:
        """
        Execute a task using AI-orchestrated tool selection and combination
        """
        logger.info(f"Executing task with AI orchestration: {task_plan.get('description', 'unknown')}")
        
        # Take a screenshot for analysis
        screenshot_path = await self.screen_agent._take_screenshot()
        
        # Build user context from memory
        user_context = {
            "preferences": self.nexus_intelligence.user_preferences,
            "recent_tasks": list(self.memory["tasks"].keys())[-5:] if self.memory.get("tasks") else []
        }
        
        # Let AI select tools and execution pattern
        orchestration = await self.nexus_intelligence.select_tools_for_task(
            task_plan.get("description", ""), 
            screenshot_path,
            user_context
        )
        
        logger.info(f"AI selected tools: {orchestration.get('selected_tools', [])}")
        
        # Check AI confidence level
        confidence = orchestration.get("confidence", 0)
        if confidence < 0.5:
            # Low confidence - ask for user confirmation if appropriate
            should_proceed = await self.nexus_intelligence.ask_user_for_clarification(
                f"I'm not very confident ({confidence:.2f}) about my approach to this task. My plan is to use: " +
                ", ".join(orchestration.get("selected_tools", [])) + 
                ". Should I proceed with this approach?",
                options=["Yes", "No, try a different approach", "Let me specify which tools to use"],
                task_context=task_plan.get("description", "")
            )
            
            if should_proceed.lower().startswith("no"):
                # Try again with a different approach
                logger.info("User requested different approach, re-selecting tools...")
                orchestration = await self.nexus_intelligence.select_tools_for_task(
                    task_plan.get("description", "") + " [ALTERNATIVE APPROACH NEEDED]", 
                    screenshot_path,
                    user_context
                )
            elif not should_proceed.lower().startswith("yes"):
                # User specified tools - parse their input
                specified_tools = [t.strip() for t in should_proceed.split(",") if t.strip()]
                available_tools = self.tool_registry.get_all_tools().keys()
                valid_tools = [t for t in specified_tools if t in available_tools]
                
                if valid_tools:
                    logger.info(f"Using user-specified tools: {valid_tools}")
                    # Create new pattern with user-selected tools
                    create_result = await self.nexus_intelligence.create_new_tool_combination(
                        task_plan.get("description", ""),
                        valid_tools
                    )
                    
                    if create_result.get("success", False):
                        orchestration = {
                            "selected_tools": valid_tools,
                            "execution_pattern": create_result.get("pattern", {}).get("execution_order", []),
                            "reasoning": "Using user-specified tools",
                            "confidence": 0.8  # Higher confidence with user guidance
                        }
        
        # Initialize results
        results = {
            "success": False,
            "steps_completed": 0,
            "steps_total": len(orchestration.get("execution_pattern", [])),
            "notes": [f"AI confidence in plan: {orchestration.get('confidence', 0):.2f}"]
        }
        
        # Check if there are suggested libraries to add
        suggested_libraries = orchestration.get("suggested_libraries", [])
        if suggested_libraries:
            libraries_str = ", ".join(suggested_libraries)
            logger.info(f"AI suggests additional libraries: {libraries_str}")
            results["notes"].append(f"Suggested libraries: {libraries_str}")
        
        # Check if we should create a new tool combination
        if len(orchestration.get("selected_tools", [])) > 1 and orchestration.get("confidence", 0) > 0.7:
            # See if we need to create a new reusable combination
            task_hash = str(hash(task_plan.get("description", "")))[:8]
            combination_name = f"task_{task_hash}"
            
            if combination_name not in self.tool_registry.combination_patterns:
                logger.info(f"Creating new tool combination: {combination_name}")
                
                create_result = await self.nexus_intelligence.create_new_tool_combination(
                    task_plan.get("description", ""),
                    orchestration.get("selected_tools", [])
                )
                
                if create_result.get("success", False):
                    results["notes"].append(f"Created new tool combination: {create_result.get('combination_name')}")
                    combination_name = create_result.get("combination_name")
                    
                    # Use this combination instead of individual steps if it was created successfully
                    if combination_name in self.tool_registry.combination_patterns:
                        return await self._execute_tool_combination(
                            combination_name, 
                            task_plan,
                            screenshot_path
                        )
    
        # Execute according to AI's plan
        execution_pattern = orchestration.get("execution_pattern", [])
        
        # Track elements and actions for learning
        elements_found = []
        actions_taken = []
        
        # Execute each step
        for i, step in enumerate(execution_pattern):
            step_tool = step.get("tool", "")
            step_params = step.get("parameters", {}) or {}  # Ensure we have a dict
            
            # Add screenshot path and task plan to parameters
            step_params["screenshot_path"] = screenshot_path
            step_params["task_plan"] = task_plan
            
            # Execute this tool
            logger.info(f"Executing step {i+1}/{len(execution_pattern)}: {step_tool}")
            
            try:
                step_result = await self.tool_registry.execute_tool(step_tool, **step_params)
                
                # Track success
                if not isinstance(step_result, dict) or "error" not in step_result:
                    results["steps_completed"] += 1
                    
                    # Store elements and actions for learning
                    if isinstance(step_result, dict):
                        if "elements" in step_result:
                            elements_found.extend(step_result.get("elements", []))
                        if "action" in step_result:
                            actions_taken.append(step_result.get("action"))
                        
                    results["notes"].append(f"Step {i+1}: {step_tool} - {'Success' if not isinstance(step_result, dict) or 'error' not in step_result else 'Failed'}")
                else:
                    results["notes"].append(f"Step {i+1}: {step_tool} - Failed: {step_result.get('error', 'Unknown error')}")
                    
                    # If this step was critical and failed, stop execution
                    if step.get("critical", False):
                        results["notes"].append(f"Critical step failed, stopping execution")
                        break
                
            except Exception as e:
                logger.error(f"Error in step {i+1}: {e}")
                results["notes"].append(f"Step {i+1}: {step_tool} - Error: {str(e)}")
                
                # If this step was critical and failed, stop execution
                if step.get("critical", False):
                    break
                
            # Take new screenshot if needed for next step
            if i < len(execution_pattern) - 1:
                screenshot_path = await self.screen_agent._take_screenshot()
        
        # Determine overall success
        results["success"] = results["steps_completed"] == results["steps_total"]
        
        # Learn from the outcome
        if results["success"]:
            self._learn_from_success(task_plan, elements_found, actions_taken)
        else:
            self._learn_from_failure(task_plan, self.current_attempt, 
                                    f"Completed {results['steps_completed']}/{results['steps_total']} steps")
        
        # Have AI learn from execution results
        await self.nexus_intelligence.learn_from_execution(
            task_plan.get("description", ""),
            results,
            1.0 if results["success"] else max(0.1, results["steps_completed"] / max(1, results["steps_total"])),
            f"Task {'completed successfully' if results['success'] else 'failed'}"
        )
        
        return results
        
    async def _execute_tool_combination(self, combination_name, task_plan, screenshot_path):
        """Execute a predefined tool combination pattern"""
        logger.info(f"Executing tool combination: {combination_name}")
        
        # Execute the combination
        result = await self.tool_registry.execute_combination(
            combination_name,
            screenshot_path=screenshot_path,
            task_plan=task_plan
        )
        
        # Format the result for compatibility with execute_task
        success = "error" not in result and not any(
            "error" in step.get("result", {}) for step in result.get("steps", [])
            if step.get("critical", False)
        )
        
        formatted_result = {
            "success": success,
            "steps_completed": sum(1 for step in result.get("steps", []) 
                                if not isinstance(step.get("result", {}), dict) or "error" not in step.get("result", {})),
            "steps_total": len(result.get("steps", [])),
            "notes": [f"Used combination pattern: {combination_name}"]
        }
        
        # Extract elements and actions for learning
        elements_found = []
        actions_taken = []
        
        for step in result.get("steps", []):
            step_result = step.get("result", {})
            if isinstance(step_result, dict):
                if "elements" in step_result:
                    elements_found.extend(step_result.get("elements", []))
                if "action" in step_result:
                    actions_taken.append(step_result.get("action"))
        
        # Learn from the outcome
        if formatted_result["success"]:
            self._learn_from_success(task_plan, elements_found, actions_taken)
        else:
            self._learn_from_failure(task_plan, self.current_attempt, 
                                    f"Combination execution failed: {result.get('error', 'Unknown error')}")
        
        # Have AI learn from execution results
        await self.nexus_intelligence.learn_from_execution(
            task_plan.get("description", ""),
            formatted_result,
            1.0 if formatted_result["success"] else 0.3,
            f"Combination {combination_name} {'succeeded' if formatted_result['success'] else 'failed'}"
        )
        
        return formatted_result

    # ... existing code ...

    async def _execute_tesseract_ocr(self, screenshot_path, task_plan=None, **kwargs):
        """Execute Tesseract OCR on a screenshot"""
        logger.info(f"Executing Tesseract OCR on screenshot: {screenshot_path}")
        
        try:
            import pytesseract
            from PIL import Image
            
            # Perform OCR
            img = Image.open(screenshot_path)
            ocr_text = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Process OCR results to extract text elements
            text_elements = []
            for i in range(len(ocr_text['text'])):
                if not ocr_text['text'][i].strip():
                    continue
                    
                text_element = {
                    "id": f"ocr_{i}",
                    "type": "text",
                    "text": ocr_text['text'][i],
                    "position": {
                        "x": ocr_text['left'][i],
                        "y": ocr_text['top'][i],
                        "width": ocr_text['width'][i],
                        "height": ocr_text['height'][i]
                    },
                    "confidence": float(ocr_text['conf'][i])/100 if ocr_text['conf'][i] > 0 else 0.5,
                    "source": "tesseract_ocr"
                }
                
                text_elements.append(text_element)
            
            # Also include full text for context
            full_text = pytesseract.image_to_string(img)
            
            return {
                "elements": text_elements,
                "full_text": full_text,
                "count": len(text_elements)
            }
        except Exception as e:
            logger.error(f"Error executing Tesseract OCR: {e}")
            return {"error": str(e)}

    async def _execute_opencv_analysis(self, screenshot_path, task_plan=None, **kwargs):
        """Execute OpenCV analysis on a screenshot"""
        logger.info(f"Executing OpenCV analysis on screenshot: {screenshot_path}")
        
        try:
            import cv2
            import numpy as np
            
            # Load the screenshot image
            img = cv2.imread(screenshot_path)
            if img is None:
                logger.error(f"Failed to load screenshot from {screenshot_path}")
                return {"error": "Failed to load screenshot"}
                
            # Get image dimensions
            height, width = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Initialize results
            ui_elements = []
            
            # 1. Detect rectangles that might be buttons or input fields
            # Look for rectangular shapes that might be UI elements
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            for i, contour in enumerate(contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small or very large rectangles
                if w < 20 or h < 10 or w > width * 0.9 or h > height * 0.9:
                    continue
                
                # Approximate contour shape
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Check if it's approximately rectangular (4 points)
                if len(approx) == 4:
                    # This could be a button, input field, or other UI element
                    aspect_ratio = float(w) / h
                    
                    # Determine element type based on aspect ratio
                    element_type = "unknown"
                    if 2.5 < aspect_ratio < 8:
                        element_type = "input_field"  # Long rectangle likely an input field
                    elif 1 < aspect_ratio < 2.5:
                        element_type = "button"  # Squarish likely a button
                    
                    ui_elements.append({
                        "id": f"opencv_element_{i}",
                        "type": element_type,
                        "text": "",  # OpenCV doesn't extract text
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h
                        },
                        "confidence": 0.7,
                        "source": "opencv"
                    })
            
            # 2. Find search bars using enhanced detection
            search_bars = self._enhanced_find_search_bars(img_rgb, height, width)
            if search_bars:
                ui_elements.extend(search_bars)
                
            # 3. Look for icons and special UI features
            # Look for circular elements (could be buttons or icons)
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i, circle in enumerate(circles[0, :]):
                    center_x, center_y, radius = circle
                    ui_elements.append({
                        "id": f"opencv_circle_{i}",
                        "type": "icon",
                        "text": "",
                        "position": {
                            "x": int(center_x - radius),
                            "y": int(center_y - radius),
                            "width": int(radius * 2),
                            "height": int(radius * 2)
                        },
                        "confidence": 0.6,
                        "source": "opencv"
                    })
            
            return {
                "elements": ui_elements,
                "count": len(ui_elements)
            }
            
        except Exception as e:
            logger.error(f"Error executing OpenCV analysis: {e}")
            return {"error": str(e)}

    async def _execute_duckduckgo_search(self, query, task_plan=None, **kwargs):
        """Execute a DuckDuckGo search"""
        logger.info(f"Executing DuckDuckGo search for query: {query}")
        
        try:
            # Import here to avoid dependencies
            from duckduckgo_search import DDGS
            
            # Initialize the search client
            ddgs = DDGS()
            
            # Execute the search
            results = []
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", "")
                })
            return {"results": results}
            
        except ImportError:
            logger.warning("DuckDuckGo search package not available, using fallback")
            # Fallback method with basic requests
            try:
                import requests
                from bs4 import BeautifulSoup
                
                # Format search URL
                search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
                
                # Execute search request
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(search_url, headers=headers)
                
                if response.status_code != 200:
                    return {"error": f"Search failed with status code {response.status_code}"}
                
                # Parse HTML response
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract search results
                results = []
                for result in soup.select(".result"):
                    title_element = result.select_one(".result__title")
                    snippet_element = result.select_one(".result__snippet")
                    
                    title = title_element.get_text().strip() if title_element else ""
                    snippet = snippet_element.get_text().strip() if snippet_element else ""
                    href = title_element.a["href"] if title_element and title_element.a else ""
                    
                    results.append({
                        "title": title,
                        "body": snippet,
                        "href": href
                    })
                    
                    # Limit to 3 results
                    if len(results) >= 3:
                        break
                        
                return {"results": results}
                
            except Exception as e:
                logger.error(f"Fallback search failed: {e}")
                return {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return {"error": str(e)}
    
    def _enhanced_find_search_bars(self, img_rgb, height, width):
        """
        Enhanced search element detection that combines OpenCV template matching with AI vision
        """
        search_elements = []
        
        try:
            import cv2
            import numpy as np
            
            # Method 1: Template matching for magnifying glass icons
            # Load the magnifying glass template
            template_path = os.path.join(os.path.dirname(__file__), "templates", "search_icon.png")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.7
                    locations = np.where(result >= threshold)
                    
                    for pt in zip(*locations[::-1]):
                        search_elements.append({
                            "id": f"search_icon_{len(search_elements)}",
                            "type": "search_bar",
                            "text": "",
                            "position": {
                                "x": int(pt[0]),
                                "y": int(pt[1]),
                                "width": template.shape[1],
                                "height": template.shape[0]
                            },
                            "confidence": float(result[pt[1], pt[0]]),
                            "source": "template_matching"
                        })
            
            # Method 2: Check for search-box like rectangles
            # Look for rectangular shapes that might be search boxes
            img_edges = cv2.Canny(img_rgb, 50, 150)
            contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter potential search boxes by aspect ratio and size
                if 3 < w/h < 10 and w > width * 0.1 and h > 20:
                    search_elements.append({
                        "id": f"search_box_{len(search_elements)}",
                        "type": "search_bar",
                        "text": "",
                        "position": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h)
                        },
                        "confidence": 0.6,
                        "source": "contour_detection"
                    })
            
        except Exception as e:
            logger.error(f"Error in search bar detection: {e}")
        
        return search_elements

    async def initialize(self):
        """Initialize the NEXUS demo with all required components"""
        logger.info("Initializing Ollama integration...")
        await self.ollama.initialize()
        
        # Initialize tool registry with all available tools
        self.tool_registry = NexusToolRegistry()
        
        # Initialize services
        self.research_service = ResearchService()
        
        # Register core tools
        self.tool_registry.register_tool(
            "tesseract_ocr",
            self._execute_tesseract_ocr,
            categories=["text_recognition", "ocr"],
            description="Tesseract OCR for text recognition from images"
        )
        
        self.tool_registry.register_tool(
            "opencv_analysis",
            self._execute_opencv_analysis,
            categories=["image_analysis", "object_detection"],
            description="OpenCV for image analysis and UI element detection"
        )
        
        self.tool_registry.register_tool(
            "duckduckgo_search",
            self._execute_duckduckgo_search,
            categories=["web_search", "information_retrieval"],
            description="DuckDuckGo search for web-based information retrieval without API keys"
        )
        
        self.tool_registry.register_tool(
            "enhanced_search_detection",
            lambda screenshot_path, **kwargs: {"elements": self._enhanced_find_search_bars(
                cv2.cvtColor(cv2.imread(screenshot_path), cv2.COLOR_BGR2RGB), 
                *cv2.imread(screenshot_path).shape[:2])},
            categories=["ui_analysis", "search"],
            description="Advanced detection of search elements using template matching and computer vision"
        )
        
        # Register research service tools
        await self.research_service.register_with_nexus(self.tool_registry)
        logger.info("Research service registered with NEXUS")
        
        # Try to register additional tools if libraries are available
        try:
            import easyocr
            self.tool_registry.register_tool(
                "easyocr_recognition",
                self._execute_easyocr,
                categories=["text_recognition", "multilingual"],
                description="Extract text from images with better multilingual support"
            )
        except ImportError:
            logger.info("EasyOCR not available, skipping registration")
        
        try:
            import mediapipe as mp
            self.tool_registry.register_tool(
                "mediapipe_detection",
                self._execute_mediapipe_detection,
                categories=["gesture_recognition", "face_detection", "pose_detection"],
                description="Detect faces, hands, and body poses using MediaPipe"
            )
        except ImportError:
            logger.info("MediaPipe not available, skipping registration")
            
        try:
            import tensorflow as tf
            self.tool_registry.register_tool(
                "tensorflow_object_detection",
                self._execute_tensorflow_detection,
                categories=["object_detection", "image_classification"],
                description="Detect and classify objects in images using TensorFlow"
            )
        except ImportError:
            logger.info("TensorFlow not available, skipping registration")
            
        # Initialize the AI intelligence layer
        self.nexus_intelligence = NexusIntelligence(self.ollama, self.tool_registry)
        
        logger.info(f"NEXUS tools initialized with {len(self.tool_registry.tools)} tools")
        logger.info("NEXUS Autonomous Demo components initialized")
    
    # ... existing code ...

class NexusToolRegistry:
    """
    Dynamic tool registry that allows AI to discover, combine and orchestrate tools
    """
    def __init__(self):
        self.tools = {}
        self.tool_categories = {}
        self.combination_patterns = {}
        self.execution_history = []
        
    def register_tool(self, tool_name, tool_function, categories=None, description=None, requires=None):
        """Register a tool with the system"""
        self.tools[tool_name] = {
            "function": tool_function,
            "categories": categories or [],
            "description": description or "",
            "requires": requires or [],
            "success_count": 0,
            "usage_count": 0
        }
        
        # Add to categories for quick lookup
        for category in categories or []:
            self.tool_categories.setdefault(category, []).append(tool_name)
            
        logger.info(f"Registered tool: {tool_name} in categories: {categories}")
        return True
        
    def get_all_tools(self):
        """Get all registered tools with their metadata"""
        return {name: {k: v for k, v in data.items() if k != "function"} 
                for name, data in self.tools.items()}
                
    def get_tools_by_category(self, category):
        """Get all tools in a specific category"""
        return self.tool_categories.get(category, [])
        
    async def execute_tool(self, tool_name, **kwargs):
        """Execute a specific tool"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
            
        try:
            tool_data = self.tools[tool_name]
            tool_function = tool_data["function"]
            
            # Execute the tool
            start_time = time.time()
            result = await tool_function(**kwargs) if asyncio.iscoroutinefunction(tool_function) else tool_function(**kwargs)
            execution_time = time.time() - start_time
            
            # Update statistics
            tool_data["usage_count"] += 1
            if not isinstance(result, dict) or "error" not in result:
                tool_data["success_count"] += 1
                
            # Record execution history
            self.execution_history.append({
                "tool": tool_name,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "success": not isinstance(result, dict) or "error" not in result,
                "parameters": {k: str(v)[:100] for k, v in kwargs.items()}  # Truncate long values
            })
            
            # Limit history size
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
            
    async def register_combination_pattern(self, name, tools, execution_order, description=None):
        """Register a pattern for combining multiple tools"""
        self.combination_patterns[name] = {
            "tools": tools,
            "execution_order": execution_order,
            "description": description or "",
            "success_count": 0,
            "usage_count": 0,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Registered new tool combination pattern: {name}")
        return True
        
    async def execute_combination(self, combination_name, **kwargs):
        """Execute a combination of tools in the specified order"""
        if combination_name not in self.combination_patterns:
            return {"error": f"Combination '{combination_name}' not found"}
            
        try:
            pattern = self.combination_patterns[combination_name]
            result = {"steps": [], "combination": combination_name}
            
            # Execute each tool in the specified order
            for i, step in enumerate(pattern["execution_order"]):
                tool_name = step["tool"]
                
                # Build parameters, which can include results from previous steps
                params = {}
                for param_name, param_source in step.get("parameters", {}).items():
                    if isinstance(param_source, str) and param_source.startswith("$result."):
                        # Extract from previous result
                        parts = param_source[8:].split(".")
                        value = result
                        for part in parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        params[param_name] = value
                    else:
                        # Use directly or from kwargs
                        params[param_name] = kwargs.get(param_name, param_source)
                
                # Execute this step
                logger.info(f"Executing combination step {i+1}: {tool_name}")
                step_result = await self.execute_tool(tool_name, **params)
                result["steps"].append({
                    "tool": tool_name,
                    "result": step_result
                })
                
                # If this step has an error and it's marked as critical, abort
                if isinstance(step_result, dict) and "error" in step_result and step.get("critical", False):
                    result["error"] = f"Critical step '{tool_name}' failed: {step_result['error']}"
                    result["aborted"] = True
                    break
            
            # Update combination statistics
            pattern["usage_count"] += 1
            if "error" not in result:
                pattern["success_count"] += 1
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing combination {combination_name}: {e}")
            return {"error": str(e)}
            
    def save_registry_state(self, file_path="tool_registry.json"):
        """Save the current registry state to disk"""
        try:
            # Create a serializable copy without function references
            state = {
                "tools": {name: {k: v for k, v in data.items() if k != "function"} 
                         for name, data in self.tools.items()},
                "tool_categories": self.tool_categories,
                "combination_patterns": self.combination_patterns,
                "execution_history": self.execution_history
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
                
            return {"success": True}
        except Exception as e:
            logger.error(f"Error saving registry state: {e}")
            return {"error": str(e)}
            
    def load_registry_state(self, file_path="tool_registry.json"):
        """Load registry state from disk (functions must be re-registered)"""
        try:
            if not os.path.exists(file_path):
                return {"error": "Registry state file not found"}
                
            with open(file_path, 'r') as f:
                state = json.load(f)
                
            # Restore what we can (tool functions must be registered separately)
            self.tool_categories = state.get("tool_categories", {})
            self.combination_patterns = state.get("combination_patterns", {})
            self.execution_history = state.get("execution_history", [])
            
            # Update tool metadata (but not functions)
            for name, data in state.get("tools", {}).items():
                if name in self.tools:
                    # Preserve the function but update metadata
                    function = self.tools[name]["function"]
                    self.tools[name] = data
                    self.tools[name]["function"] = function
                    
            return {"success": True}
        except Exception as e:
            logger.error(f"Error loading registry state: {e}")
            return {"error": str(e)}

class NexusIntelligence:
    """AI layer that makes decisions about tool usage and combinations"""
    
    def __init__(self, ollama_client, tool_registry):
        self.ollama = ollama_client
        self.registry = tool_registry
        self.learning_history = []
        self.user_preferences = {}
        
    async def select_tools_for_task(self, task_description, screenshot_path=None, user_context=None):
        """
        Let AI select the appropriate tools and execution pattern for a task
        """
        # 1. Get information about available tools
        all_tools = self.registry.get_all_tools()
        tools_by_category = {cat: self.registry.get_tools_by_category(cat) 
                            for cat in set(sum([t.get("categories", []) for t in all_tools.values()], []))}
        
        # 2. Create selection prompt
        tools_info = "\n".join([f"- {name}: {data['description']}" 
                              for name, data in all_tools.items()])
                              
        categories_info = "\n".join([f"- {cat}: {', '.join(tools)}" 
                                  for cat, tools in tools_by_category.items()])
        
        # Include user context and preferences if available
        user_context_str = ""
        if user_context:
            user_context_str = f"\nUser context: {json.dumps(user_context, indent=2)}"
            
        # Include recent learning history for adaptation
        learning_str = ""
        if self.learning_history:
            recent_learnings = self.learning_history[-3:]  # Last 3 learnings
            learning_str = "\nRecent learning experiences:\n" + "\n".join([
                f"- Task: {item['task']}, Success: {item['success_rating']}"
                for item in recent_learnings
            ])
        
        prompt = f"""
        I need to select the right tools and execution pattern for this task:
        
        TASK: {task_description}
        
        Available tools:
        {tools_info}
        
        Tool categories:
        {categories_info}
        {user_context_str}
        {learning_str}
        
        Please analyze the task and any visible content in the screenshot to determine:
        
        1. Which tools would be most effective for this task
        2. What sequence and combination pattern would work best
        3. How the output of each tool should feed into others
        
        Return your analysis as JSON with:
        {{
            "selected_tools": ["tool1", "tool2", ...],
            "execution_pattern": [
                {{
                    "step": 1,
                    "tool": "tool1",
                    "parameters": {{...}},
                    "description": "why this tool is needed",
                    "critical": true/false
                }},
                ...
            ],
            "reasoning": "explanation for your choices",
            "confidence": 0.0-1.0,
            "suggested_libraries": [] // Optional libraries that could improve this task
        }}
        """
        
        # 3. Get AI's selection
        if screenshot_path and os.path.exists(screenshot_path):
            selection_result = await self.ollama.generate_text(
                prompt,
                model="llava-next:7b",  # Vision-capable model
                system_prompt="You are the NEXUS intelligence orchestration system. Select the best tools for the task.",
                image_path=screenshot_path
            )
        else:
            selection_result = await self.ollama.generate_text(
                prompt,
                system_prompt="You are the NEXUS intelligence orchestration system. Select the best tools for the task."
            )
        
        # 4. Parse AI's response
        try:
            if '{' in selection_result and '}' in selection_result:
                json_text = selection_result[selection_result.find('{'):selection_result.rfind('}')+1]
                return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing AI tool selection response: {e}")
            logger.debug(f"Raw AI response: {selection_result[:500]}...")
            
        # 5. Fallback if JSON parsing failed
        return {
            "selected_tools": self._get_default_tools_for_task(task_description),
            "execution_pattern": [{"step": 1, "tool": "tesseract_ocr", "parameters": {}, "critical": True}],
            "reasoning": "Failed to parse AI response, using default tools",
            "confidence": 0.3
        }
        
    async def create_new_tool_combination(self, task_description, tools_to_combine, previous_results=None):
        """
        Let AI create a new combination pattern for a set of tools
        """
        # Get detailed info about the tools
        tools_info = {}
        for tool_name in tools_to_combine:
            if tool_name in self.registry.tools:
                tool_data = self.registry.tools[tool_name]
                tools_info[tool_name] = {
                    "description": tool_data.get("description", ""),
                    "categories": tool_data.get("categories", []),
                    "success_rate": tool_data.get("success_count", 0) / max(1, tool_data.get("usage_count", 1))
                }
        
        # Include previous results if available
        previous_results_str = ""
        if previous_results:
            previous_results_str = f"\nPrevious execution results:\n{json.dumps(previous_results, indent=2)}"
        
        prompt = f"""
        I need to create a new tool combination pattern for this task:
        
        TASK: {task_description}
        
        Tools to combine:
        {json.dumps(tools_info, indent=2)}
        {previous_results_str}
        
        Please create an execution pattern that:
        1. Uses these tools in the most effective sequence
        2. Passes data between tools appropriately
        3. Handles potential failures gracefully
        
        Return your pattern as JSON with:
        {{
            "name": "descriptive_name_for_this_pattern",
            "description": "what this pattern does",
            "execution_order": [
                {{
                    "step": 1,
                    "tool": "tool1",
                    "parameters": {{...}},
                    "description": "why this tool is needed",
                    "critical": true/false
                }},
                ...
            ]
        }}
        """
        
        # Get AI's combination pattern
        pattern_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS tool orchestration system. Create optimal tool combinations."
        )
        
        # Parse AI's response
        try:
            if '{' in pattern_result and '}' in pattern_result:
                json_text = pattern_result[pattern_result.find('{'):pattern_result.rfind('}')+1]
                pattern = json.loads(json_text)
                
                # Register this combination
                name = pattern.get("name", f"combination_{int(time.time())}")
                await self.registry.register_combination_pattern(
                    name,
                    tools_to_combine,
                    pattern.get("execution_order", []),
                    pattern.get("description", "")
                )
                
                return {"success": True, "combination_name": name, "pattern": pattern}
        except Exception as e:
            logger.error(f"Error creating combination: {e}")
            
        return {"success": False, "error": "Failed to create combination pattern"}
        
    async def learn_from_execution(self, task, execution_result, success_rating, user_feedback=None):
        """Learn from execution results to improve future tool selection"""
        self.learning_history.append({
            "task": task,
            "tools_used": [step.get("tool") for step in execution_result.get("steps", [])] if "steps" in execution_result else [],
            "success_rating": success_rating,
            "user_feedback": user_feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
            
        # Store learning insights if provided
        if user_feedback and isinstance(user_feedback, str):
            # Extract potential insights from user feedback
            await self._extract_insights_from_feedback(task, user_feedback)
            
        # AI analysis of learning opportunity
        if success_rating < 0.7 and user_feedback:
            # Ask AI to suggest improvements
            await self._analyze_failure_for_learning(task, execution_result, user_feedback)
            
    async def ask_user_for_clarification(self, question, options=None, task_context=None):
        """
        Formulate a question to ask the user for clarification when AI is uncertain
        Returns the user's response
        """
        # Format the question with context if provided
        formatted_question = question
        if task_context:
            formatted_question = f"About task: {task_context}\n\n{question}"
            
        # Add options if provided
        if options:
            formatted_question += "\n\nOptions:\n" + "\n".join([f"- {opt}" for opt in options])
            
        # In a real system, this would display a UI prompt
        # For now, we'll simulate by printing to console
        print(f"\n🤔 NEXUS needs clarification: {formatted_question}")
        
        # Get user input
        user_response = input("Your answer: ")
        
        # Record this interaction in learning history
        self.learning_history.append({
            "type": "clarification",
            "question": question,
            "user_response": user_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return user_response
        
    async def suggest_new_libraries(self, task_description, current_capabilities):
        """
        Have AI suggest new libraries or tools that could enhance capabilities
        """
        prompt = f"""
        Based on this task and our current capabilities, suggest new libraries or tools that could enhance NEXUS:
        
        TASK: {task_description}
        
        Current capabilities:
        {json.dumps(current_capabilities, indent=2)}
        
        What new libraries, APIs, or tools could help NEXUS better accomplish this task?
        Focus on free, open-source options where possible.
        
        For each suggestion, explain:
        1. What the library/tool does
        2. How it would help with this task
        3. How complex it would be to integrate
        4. Example Python code for how it might be used
        
        Return suggestions as JSON.
        """
        
        suggestions = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS intelligence enhancement advisor."
        )
        
        try:
            if '{' in suggestions and '}' in suggestions:
                json_text = suggestions[suggestions.find('{'):suggestions.rfind('}')+1]
                return json.loads(json_text)
        except:
            pass
            
        # Return raw text if JSON parsing failed
        return {"raw_suggestions": suggestions}
    
    async def _extract_insights_from_feedback(self, task, user_feedback):
        """Extract learning insights from user feedback"""
        prompt = f"""
        Extract key learning insights from this user feedback:
        
        TASK: {task}
        USER FEEDBACK: {user_feedback}
        
        Extract:
        1. User preferences
        2. Task patterns
        3. Tool effectiveness
        4. Improvement suggestions
        
        Return insights as JSON.
        """
        
        insights = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS learning system."
        )
        
        try:
            if '{' in insights and '}' in insights:
                json_text = insights[insights.find('{'):insights.rfind('}')+1]
                insights_data = json.loads(json_text)
                
                # Update user preferences if found
                if "user_preferences" in insights_data:
                    for pref, value in insights_data["user_preferences"].items():
                        self.user_preferences[pref] = value
                        
                return insights_data
        except:
            pass
            
        return {"extracted_insights": "Failed to extract structured insights"}
            
    async def _analyze_failure_for_learning(self, task, execution_result, user_feedback):
        """Have AI analyze a failure for learning opportunities"""
        prompt = f"""
        I need to learn from this execution failure:
        
        TASK: {task}
        
        Execution Result: 
        {json.dumps(execution_result, indent=2)}
        
        User Feedback:
        {user_feedback}
        
        Please analyze what went wrong and suggest improvements:
        1. Were the wrong tools selected?
        2. Was the execution order incorrect?
        3. Should different parameters have been used?
        4. Is there a gap in our tool capabilities?
        
        Return your analysis as JSON.
        """
        
        analysis = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS learning and improvement system."
        )
        
        # We could use this analysis to automatically improve the system in the future
        logger.info(f"Learning opportunity analysis: {analysis[:100]}...")
        
        # Add to learning history
        self.learning_history.append({
            "type": "failure_analysis",
            "task": task,
            "analysis": analysis[:500],  # Store first 500 chars
            "timestamp": datetime.now().isoformat()
        })
        
    def _get_default_tools_for_task(self, task_description):
        """Fallback method to select default tools based on task keywords"""
        task_lower = task_description.lower()
        
        if "search" in task_lower:
            return ["tesseract_ocr", "enhanced_search_detection", "duckduckgo_search"]
        elif "image" in task_lower or "picture" in task_lower or "photo" in task_lower:
            return ["opencv_analysis", "tesseract_ocr"]
        elif any(word in task_lower for word in ["browser", "web", "website", "internet"]):
            return ["tesseract_ocr", "duckduckgo_search"]
        else:
            return ["tesseract_ocr", "opencv_analysis"]  # Default tools

class NexusAutonomousDemo:
    # ... existing code ...

    async def run_demo(self):
        """Run the NEXUS autonomous agent demo"""
        # Initialize the demo
        await self.initialize()
        
        print("NEXUS Autonomous AI Demo initialized successfully")
        print("AI is ready to process natural language commands")
        print("Screen size detected:", self.screen_size)
        print("----------------------------------------------------\n")
        
        # Main interaction loop
        running = True
        while running:
            try:
                # Get user command
                command = input("Enter a command (or 'quit' to exit):\n> ")
                
                if command.lower() == "quit":
                    running = False
                    print("Thank you for using NEXUS Autonomous AI Demo")
                    continue
                
                # Process the command
                if command:
                    # Parse to get task understanding
                    task_plan = await self.understand_task(command)
                    
                    print("\nUnderstanding task:")
                    print("Target application:", task_plan.get("application", "unknown"))
                    print("Number of steps:", len(task_plan.get("steps", [])))
                    
                    # Execute the task
                    results = await self.execute_task(task_plan)
                    
                    # Report results
                    if results["success"]:
                        print("\n✅ Task completed successfully!")
                    else:
                        print(f"\n⚠️ Task partially completed ({results['steps_completed']}/{results['steps_total']} steps)")
                    
                    for note in results.get("notes", []):
                        print(f"- {note}")
                else:
                    print("\n❌ Sorry, I couldn't understand how to accomplish that task.")
            
            except KeyboardInterrupt:
                print("\nTask canceled by user")
                running = False
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(NexusAutonomousDemo().run_demo())
    except KeyboardInterrupt:
        print("\nDemo canceled by user")
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        import traceback
        traceback.print_exc()
