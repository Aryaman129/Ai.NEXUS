"""
Automation AI for NEXUS
Specialized AI for UI interactions, system actions, and automation tasks
"""
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import NEXUS modules
from .shared_memory import SharedMemory
from ..modules.vision_ui_automation import VisionUIAutomation

logger = logging.getLogger(__name__)

class AutomationAI:
    """
    Automation AI specialist that handles UI interactions, system actions,
    and automation tasks for NEXUS.
    """
    
    def __init__(self, ollama_client, memory: SharedMemory, tool_registry):
        """Initialize the Automation AI specialist"""
        self.ollama = ollama_client
        self.memory = memory
        self.tool_registry = tool_registry
        self.name = "automation"
        self.reasoning_log = []
        self.ui_automation = VisionUIAutomation()
        
        # Load automation-related tools from registry
        self.automation_tools = {}
        for tool_name, tool_info in self.tool_registry.get_all_tools().items():
            categories = tool_info.get("categories", [])
            automation_categories = ["ui_interaction", "system_action", "automation", 
                                     "keyboard", "mouse", "file_operation"]
            
            if any(cat in automation_categories for cat in categories):
                self.automation_tools[tool_name] = tool_info
        
        logger.info(f"NEXUS Automation AI initialized with {len(self.automation_tools)} automation tools")
    
    async def execute_subtask(self, subtask: Dict[str, Any], 
                            dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an automation-related subtask
        
        Args:
            subtask: The subtask definition from the Coordinator
            dependency_results: Results from dependent subtasks
            
        Returns:
            Subtask execution results
        """
        subtask_id = subtask.get("id", "unknown")
        goal = subtask.get("goal", "")
        
        logger.info(f"Automation AI executing subtask {subtask_id}: {goal}")
        
        try:
            # Log receipt of this subtask
            self.memory.log_specialist_communication(
                from_ai="coordinator",
                to_ai=self.name,
                message={"action": "received_subtask", "subtask_id": subtask_id}
            )
            
            # Plan the automation steps
            action_plan = await self._plan_automation_steps(subtask, dependency_results)
            
            # Execute the automation steps
            results = await self._execute_action_steps(action_plan, dependency_results)
            
            # Verify the actions were successful
            verification = await self._verify_actions(results, subtask)
            
            # Return a structured result
            success = verification.get("success", False)
            final_result = {
                "status": "completed",
                "success": success,
                "actions_performed": results,
                "verification": verification,
                "subtask_id": subtask_id,
                "message": f"Automation {'successful' if success else 'failed'} for subtask {subtask_id}"
            }
            
            # Log completion
            self.memory.log_specialist_communication(
                from_ai=self.name,
                to_ai="coordinator",
                message={"action": "completed_subtask", "subtask_id": subtask_id, "success": success}
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in Automation AI executing subtask {subtask_id}: {e}")
            
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
    
    async def _plan_automation_steps(self, subtask: Dict[str, Any], 
                                  dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Plan the automation steps based on the subtask and dependency results"""
        goal = subtask.get("goal", "")
        
        # Get context from dependency results
        vision_context = ""
        research_context = ""
        
        if dependency_results:
            for dep_id, dep_result in dependency_results.items():
                if "findings" in dep_result:
                    # Check if it's from vision AI
                    if "extracted_text" in dep_result["findings"]:
                        vision_context += f"\nVision findings: {json.dumps(dep_result['findings'])[:500]}"
                    # Check if it's from research AI
                    elif "information" in dep_result["findings"]:
                        research_context += f"\nResearch findings: {json.dumps(dep_result['findings'])[:500]}"
        
        # Create a prompt for the AI to determine the best automation approach
        prompt = f"""
        I need to create an automation plan for this task:
        
        GOAL: {goal}
        
        Vision context: {vision_context}
        Research context: {research_context}
        
        I have the following automation tools available:
        {', '.join(self.automation_tools.keys())}
        
        For this task, determine:
        1. The sequence of UI or system actions needed to accomplish the goal
        2. What specific elements to interact with (buttons, fields, etc.)
        3. Any text input or data to enter
        4. How to verify the actions were successful
        
        Return your automation plan as JSON with these fields:
        - action_steps: array of action steps, each with:
          * action: the type of action (click, type, drag, etc.)
          * target: what to target (element description, coordinates, etc.)
          * input_data: any data to input (for text fields, etc.)
          * wait_after: milliseconds to wait after this action
        - verification: how to verify success (text to find, UI state to check)
        - fallback: alternative actions if primary actions fail
        """
        
        # Log this reasoning step
        await self._log_reasoning("automation_planning", prompt)
        
        # Get AI response for automation planning
        plan_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Automation AI specialist that plans automation steps."
        )
        
        # Extract JSON from response
        plan = self._extract_json(plan_result)
        if not plan:
            logger.warning("Failed to extract action plan from AI response, using fallback plan")
            # Provide a fallback simple plan
            plan = {
                "action_steps": [
                    {"action": "click", "target": "center of screen", "input_data": None, "wait_after": 1000}
                ],
                "verification": {"check_for": "Success indicator"},
                "fallback": {"action": "retry"}
            }
        
        logger.info(f"Created automation plan with {len(plan.get('action_steps', []))} steps")
        return plan
    
    async def _execute_action_steps(self, plan: Dict[str, Any], 
                                dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the automation steps according to the plan"""
        action_steps = plan.get("action_steps", [])
        results = {}
        
        # If no action steps specified, return empty results
        if not action_steps:
            logger.warning("No action steps to execute")
            return {"error": "No action steps in plan"}
        
        # Execute each action step
        for i, step in enumerate(action_steps):
            step_id = f"step_{i+1}"
            action = step.get("action", "")
            target = step.get("target", "")
            input_data = step.get("input_data")
            wait_after = step.get("wait_after", 1000)  # Default 1 second wait
            
            try:
                logger.info(f"Executing automation step {step_id}: {action} on {target}")
                
                # Take screenshot for visual targeting
                screenshot = await self.ui_automation.take_screenshot()
                
                # Handle different action types
                if action == "click":
                    # Try to find element by template or description
                    if target.endswith(".png") or target.endswith(".jpg"):
                        # Template matching
                        element = await self.ui_automation.find_element_by_template(
                            screenshot=screenshot,
                            template_path=target
                        )
                    else:
                        # Try to locate by text or element description
                        element = await self.ui_automation.find_element_by_description(
                            screenshot=screenshot,
                            description=target,
                            dependency_results=dependency_results
                        )
                    
                    if element and "location" in element:
                        # Click on the element
                        click_result = await self.ui_automation.click(
                            location=element["location"]
                        )
                        results[step_id] = {
                            "action": "click",
                            "target": target,
                            "result": click_result,
                            "success": click_result.get("success", False)
                        }
                    else:
                        results[step_id] = {
                            "action": "click",
                            "target": target,
                            "error": "Element not found",
                            "success": False
                        }
                        
                elif action == "type":
                    # First click on target if necessary
                    if target:
                        # Try to find element by description
                        element = await self.ui_automation.find_element_by_description(
                            screenshot=screenshot,
                            description=target,
                            dependency_results=dependency_results
                        )
                        
                        if element and "location" in element:
                            await self.ui_automation.click(location=element["location"])
                    
                    # Type the input data
                    if input_data:
                        type_result = await self.ui_automation.type_text(text=input_data)
                        results[step_id] = {
                            "action": "type",
                            "target": target,
                            "input": input_data,
                            "result": type_result,
                            "success": type_result.get("success", False)
                        }
                    else:
                        results[step_id] = {
                            "action": "type",
                            "target": target,
                            "error": "No input data provided",
                            "success": False
                        }
                        
                elif action == "press_key":
                    # Press a specific key
                    key_result = await self.ui_automation.press_key(key=target)
                    results[step_id] = {
                        "action": "press_key",
                        "key": target,
                        "result": key_result,
                        "success": key_result.get("success", False)
                    }
                    
                elif action == "drag":
                    # Drag from one location to another
                    # Target format expected as "x1,y1 to x2,y2"
                    try:
                        src, dst = target.split(" to ")
                        src_x, src_y = map(int, src.split(","))
                        dst_x, dst_y = map(int, dst.split(","))
                        
                        drag_result = await self.ui_automation.drag(
                            start_location=(src_x, src_y),
                            end_location=(dst_x, dst_y)
                        )
                        
                        results[step_id] = {
                            "action": "drag",
                            "from": src,
                            "to": dst,
                            "result": drag_result,
                            "success": drag_result.get("success", False)
                        }
                    except Exception as e:
                        results[step_id] = {
                            "action": "drag",
                            "target": target,
                            "error": f"Invalid drag target format: {str(e)}",
                            "success": False
                        }
                        
                else:
                    # Use a tool from the registry if it matches the action
                    if action in self.automation_tools:
                        tool_result = await self.tool_registry.execute_tool(
                            action,
                            target=target, 
                            input_data=input_data
                        )
                        
                        results[step_id] = {
                            "action": action,
                            "target": target,
                            "result": tool_result,
                            "success": "error" not in tool_result
                        }
                        
                        # Update tool statistics in memory
                        self.memory.update_tool_stats(
                            action, 
                            success="error" not in tool_result
                        )
                    else:
                        results[step_id] = {
                            "action": action,
                            "target": target,
                            "error": f"Unknown action type: {action}",
                            "success": False
                        }
                
                # Wait after action
                if wait_after > 0:
                    await asyncio.sleep(wait_after / 1000.0)  # Convert ms to seconds
                
            except Exception as e:
                logger.error(f"Error executing automation step {step_id}: {e}")
                results[step_id] = {
                    "action": action,
                    "target": target,
                    "error": str(e),
                    "success": False
                }
                
                # Check if we should continue after error
                if not plan.get("continue_on_error", False):
                    logger.warning(f"Stopping automation execution due to error in step {step_id}")
                    break
        
        return results
    
    async def _verify_actions(self, action_results: Dict[str, Any], 
                           subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the actions were successful"""
        goal = subtask.get("goal", "")
        
        # Take a screenshot for verification
        screenshot = await self.ui_automation.take_screenshot()
        
        # Check if all steps were successful
        all_steps_succeeded = all(
            step.get("success", False) 
            for step_id, step in action_results.items()
        )
        
        # Create a prompt for the AI to verify success
        prompt = f"""
        I need to verify if this automation task was successful:
        
        GOAL: {goal}
        
        Action results:
        {json.dumps(action_results, indent=2)}
        
        Please analyze:
        1. Did all steps execute successfully?
        2. Based on the goal, what should be visible or changed if the task succeeded?
        3. Are there any specific indicators of success or failure?
        
        Return your verification as JSON with these fields:
        - success: boolean indicating if the overall task was successful
        - verification_method: how you determined success
        - confidence: confidence in the success determination (0-1)
        - notes: any important observations
        - next_steps: recommended next steps if any
        """
        
        # Log this reasoning step
        await self._log_reasoning("automation_verification", prompt)
        
        # Get AI response for verification
        verification_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Automation AI specialist that verifies automation success."
        )
        
        # Extract JSON from response
        verification = self._extract_json(verification_result)
        if not verification:
            logger.warning("Failed to extract verification from AI response, using basic verification")
            
            # Use a basic verification based on step success
            verification = {
                "success": all_steps_succeeded,
                "verification_method": "Step success check",
                "confidence": 0.7 if all_steps_succeeded else 0.3,
                "notes": ["Verification based only on reported step success"],
                "next_steps": ["None" if all_steps_succeeded else "Retry with modified approach"]
            }
            
        # Remember the approach for future learning
        if verification.get("success", False):
            # Store successful pattern
            self.memory.learn_from_success({
                "task_type": "automation",
                "goal": goal,
                "action_sequence": [step.get("action") for step_id, step in action_results.items()],
                "verification_method": verification.get("verification_method")
            })
        else:
            # Store error pattern
            self.memory.learn_from_error({
                "error_type": "automation_failure",
                "goal": goal,
                "failed_steps": [
                    step_id for step_id, step in action_results.items() 
                    if not step.get("success", False)
                ],
                "notes": verification.get("notes", [])
            })
        
        return verification
    
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
