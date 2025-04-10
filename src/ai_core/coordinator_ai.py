"""
Coordinator AI for NEXUS
Responsible for task decomposition, specialist delegation, and result synthesis
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import shared memory
from .shared_memory import SharedMemory
# Import RAG capabilities
from .rag_engine import KnowledgeManager

logger = logging.getLogger(__name__)

class CoordinatorAI:
    """
    Coordinator AI (Task Manager) that breaks down tasks, delegates to specialists,
    and synthesizes the final results. This is the orchestration layer of NEXUS.
    """
    
    def __init__(self, ollama_client, memory: SharedMemory, tool_registry):
        """Initialize the Coordinator AI"""
        self.ollama = ollama_client
        self.memory = memory
        self.tool_registry = tool_registry
        self.name = "coordinator"
        self.reasoning_log = []
        
        # Initialize RAG knowledge manager
        self.knowledge_manager = KnowledgeManager(
            ollama_client=ollama_client,
            knowledge_base_dir="memory/coordinator_knowledge"
        )
        
        logger.info("NEXUS Coordinator AI initialized")
    
    async def process_task(self, task_description: str) -> Dict[str, Any]:
        """
        Process a user task by decomposing it, delegating to specialists,
        and synthesizing the results
        
        Args:
            task_description: The task requested by the user
            
        Returns:
            Task execution results
        """
        logger.info(f"Processing task: {task_description}")
        
        # Create a new session context
        self.memory.create_session_context(task_description)
        
        try:
            # 1. Decompose the task into subtasks
            subtasks = await self.decompose_task(task_description)
            self.memory.update_context("task_decomposition", subtasks)
            
            # 2. Execute subtasks with appropriate specialists
            results = await self.execute_subtasks(subtasks)
            
            # 3. Synthesize final result
            final_result = await self.synthesize_results(results, task_description)
            
            # 4. Store the result in memory
            self.memory.store_task_result(final_result)
            
            return final_result
        
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "steps_completed": 0,
                "steps_total": 1,
                "notes": [f"Error occurred: {str(e)}"]
            }
            self.memory.store_task_result(error_result)
            return error_result
    
    async def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a task into subtasks with specialist assignments
        
        Args:
            task_description: The task requested by the user
            
        Returns:
            List of subtasks with specialist assignments
        """
        # Retrieve related past experiences from memory to improve decomposition
        related_experiences = self.memory.get_related_experiences(task_description)
        
        # Also retrieve relevant knowledge from RAG
        relevant_knowledge = await self.knowledge_manager.retrieve_relevant_knowledge(
            query=task_description,
            category="task_patterns",
            max_results=3
        )
        
        # Format experiences as context
        experiences_context = ""
        if related_experiences:
            experiences_context = "Related past experiences:\n"
            for i, exp in enumerate(related_experiences[:3], 1):
                success = "✓" if exp.get("success", False) else "✗"
                experiences_context += f"{i}. [{success}] {exp.get('description', 'Unknown task')}\n"
        
        # Format knowledge as context
        knowledge_context = ""
        if relevant_knowledge:
            knowledge_context = "\nRelevant knowledge from past tasks:\n"
            for i, knowledge in enumerate(relevant_knowledge, 1):
                knowledge_context += f"{i}. {knowledge.get('text', '')}\n"
        
        # Create the task decomposition prompt
        base_prompt = f"""
        I need to decompose this task into logical subtasks for specialist AIs:
        
        TASK: {task_description}
        
        {experiences_context}
        {knowledge_context}
        
        The specialist AIs available are:
        1. VISION AI - For image analysis, OCR, UI element detection
        2. RESEARCH AI - For web searches, knowledge retrieval, content analysis
        3. AUTOMATION AI - For executing actions, UI interactions, system tasks
        
        For each subtask, specify:
        1. Subtask goal
        2. Which specialist should handle it 
        3. Required inputs (what it needs to know)
        4. Expected outputs (what it should produce)
        5. Dependencies (which other subtasks must complete first)
        
        Ensure the decomposition is:
        - Complete (covers all aspects of the task)
        - Properly sequenced (respects dependencies)
        - Suited to each specialist's capabilities
        
        Return as a JSON array, where each object follows this format:
        {{
            "id": "subtask-1",  
            "goal": "Subtask goal",
            "specialist": "vision|research|automation",
            "inputs": ["Description of input 1", "Description of input 2"],
            "expected_outputs": ["Description of expected output 1"],
            "dependencies": []  // IDs of subtasks this depends on
        }}
        """
        
        # Enhance the prompt with RAG knowledge if available
        prompt = await self.knowledge_manager.enhance_with_knowledge(
            original_prompt=base_prompt,
            task_description=task_description,
            relevant_categories=["task_patterns", "system_knowledge", "reasoning_patterns"]
        )
        
        # Log this reasoning step
        await self._log_reasoning("task_decomposition", prompt)
        
        # Get AI response for task decomposition
        decomposition_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Coordinator AI that decomposes tasks into logical subtasks."
        )
        
        # Extract insights from the response for learning
        await self.knowledge_manager.extract_insights(decomposition_result)
        
        # Extract JSON from response
        subtasks = self._extract_json_array(decomposition_result)
        if not subtasks:
            logger.warning("Failed to extract subtasks from AI response, using fallback decomposition")
            # Provide a fallback simple decomposition
            subtasks = self._fallback_decomposition(task_description)
        
        logger.info(f"Decomposed task into {len(subtasks)} subtasks")
        return subtasks
    
    def _extract_json_array(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from AI response text"""
        try:
            # Find JSON-like content between square brackets
            import re
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            return []
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return []
    
    def _fallback_decomposition(self, task_description: str) -> List[Dict[str, Any]]:
        """Create a fallback task decomposition when AI decomposition fails"""
        # Create a simple three-step process (analyze, research, execute)
        subtasks = [
            {
                "id": "subtask-1",
                "goal": f"Analyze visual elements related to: {task_description}",
                "specialist": "vision",
                "inputs": [task_description],
                "expected_outputs": ["Identified UI elements and text"],
                "dependencies": []
            },
            {
                "id": "subtask-2",
                "goal": f"Research information needed for: {task_description}",
                "specialist": "research",
                "inputs": [task_description],
                "expected_outputs": ["Relevant information for task execution"],
                "dependencies": []
            },
            {
                "id": "subtask-3",
                "goal": f"Execute actions for: {task_description}",
                "specialist": "automation",
                "inputs": [task_description, "Results from subtask-1", "Results from subtask-2"],
                "expected_outputs": ["Task execution results"],
                "dependencies": ["subtask-1", "subtask-2"]
            }
        ]
        return subtasks
    
    async def execute_subtasks(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute subtasks by delegating to appropriate specialists in proper order
        
        Args:
            subtasks: List of subtasks with dependencies
            
        Returns:
            Results of all subtasks
        """
        results = {}
        executed_subtasks = set()
        all_subtasks = set(subtask["id"] for subtask in subtasks)
        
        # Record active specialists
        active_specialists = set()
        for subtask in subtasks:
            specialist = subtask.get("specialist")
            if specialist:
                active_specialists.add(specialist)
        self.memory.update_context("active_specialists", list(active_specialists))
        
        # Execute subtasks in dependency order until all are executed
        while executed_subtasks != all_subtasks:
            for subtask in subtasks:
                subtask_id = subtask["id"]
                
                # Skip already executed subtasks
                if subtask_id in executed_subtasks:
                    continue
                
                # Check if dependencies are met
                dependencies = set(subtask.get("dependencies", []))
                if not dependencies.issubset(executed_subtasks):
                    logger.info(f"Skipping subtask {subtask_id} as dependencies are not met yet")
                    continue
                
                # Execute the subtask with the appropriate specialist
                specialist = subtask.get("specialist")
                if not specialist:
                    logger.warning(f"No specialist assigned for subtask {subtask_id}")
                    continue
                
                # Get inputs from dependencies
                input_data = {
                    "subtask": subtask,
                    "task_description": self.memory.get_context("current_task").get("description", ""),
                    "dependency_results": {dep: results.get(dep) for dep in subtask.get("dependencies", [])}
                }
                
                # Delegate to specialist (this will be implemented by callers)
                logger.info(f"Delegating subtask {subtask_id} to {specialist} specialist")
                
                # This will be a point where an actual message is sent to the specialist
                self.memory.log_specialist_communication(
                    from_ai=self.name,
                    to_ai=specialist,
                    message={"action": "execute_subtask", "subtask": subtask}
                )
                
                # This is a placeholder - in production this would communicate with actual specialist AIs
                # The actual execution will depend on how the specialists are implemented and integrated
                subtask_result = {
                    "status": "pending",
                    "message": f"Subtask {subtask_id} delegated to {specialist} specialist"
                }
                
                # Store the result
                results[subtask_id] = subtask_result
                executed_subtasks.add(subtask_id)
                
                # Update intermediate results in context
                self.memory.update_context("intermediate_results", results)
            
            # If we didn't execute any new subtasks in this iteration,
            # there might be a dependency cycle
            if len(executed_subtasks) < len(all_subtasks):
                remaining = all_subtasks - executed_subtasks
                logger.warning(f"Possible dependency cycle detected. Remaining subtasks: {remaining}")
                
                # Force execution of remaining subtasks
                for subtask in subtasks:
                    if subtask["id"] in remaining:
                        # Log the issue
                        logger.warning(f"Forcing execution of subtask {subtask['id']} due to possible dependency cycle")
                        
                        # Placeholder for forced execution
                        results[subtask["id"]] = {
                            "status": "forced",
                            "message": f"Subtask {subtask['id']} force-executed due to dependency cycle"
                        }
                        executed_subtasks.add(subtask["id"])
                
                # Update intermediate results in context
                self.memory.update_context("intermediate_results", results)
                
                # If we're still stuck, break to avoid infinite loop
                if len(executed_subtasks) < len(all_subtasks):
                    logger.error("Cannot resolve subtask dependencies. Breaking execution.")
                    break
        
        return results
    
    async def synthesize_results(self, subtask_results: Dict[str, Any], 
                                original_task: str) -> Dict[str, Any]:
        """
        Synthesize the final result from subtask results
        
        Args:
            subtask_results: Results of all subtasks
            original_task: Original task description
            
        Returns:
            Synthesized final result
        """
        # Create a comprehensive prompt for result synthesis
        base_prompt = f"""
        I need to synthesize the results of multiple subtasks into a final result.
        
        Original task: {original_task}
        
        Subtask results:
        {json.dumps(subtask_results, indent=2)}
        
        Please analyze these results and:
        1. Determine if the overall task was successful
        2. Summarize what was accomplished
        3. Note any issues or partial successes
        4. Identify what learning can be applied for future tasks
        
        Return your synthesis as JSON with these fields:
        - success: boolean indicating overall success
        - summary: brief summary of what was accomplished
        - steps_completed: integer count of completed steps
        - steps_total: total number of steps
        - notes: array of important notes about the execution
        - learning_insights: what can be learned from this execution
        """
        
        # Enhance the prompt with RAG knowledge if available
        prompt = await self.knowledge_manager.enhance_with_knowledge(
            original_prompt=base_prompt,
            task_description=original_task,
            relevant_categories=["task_patterns", "error_patterns"]
        )
        
        # Log this reasoning step
        await self._log_reasoning("result_synthesis", prompt)
        
        # Get AI response for synthesis
        synthesis_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Coordinator AI that synthesizes task results."
        )
        
        # Extract JSON from response
        try:
            # Find JSON-like content between curly braces
            import re
            json_match = re.search(r'\{[\s\S]*\}', synthesis_result)
            if json_match:
                result_json = json.loads(json_match.group(0))
                
                # Ensure minimal fields are present
                final_result = {
                    "success": result_json.get("success", False),
                    "summary": result_json.get("summary", "Task execution completed"),
                    "steps_completed": result_json.get("steps_completed", 0),
                    "steps_total": result_json.get("steps_total", len(subtask_results)),
                    "notes": result_json.get("notes", []),
                    "learning_insights": result_json.get("learning_insights", [])
                }
                
                # Record learning if successful
                if final_result["success"]:
                    learning_pattern = {
                        "task_type": self._categorize_task(original_task),
                        "subtasks": [s["goal"] for s in self.memory.get_context("task_decomposition")],
                        "specialists_used": self.memory.get_context("active_specialists"),
                        "learning_insights": final_result.get("learning_insights", [])
                    }
                    self.memory.learn_from_success(learning_pattern)
                    
                    # Also store in RAG knowledge manager
                    await self.knowledge_manager.learn_from_experience(
                        experience={
                            "description": f"Successfully completed task: {original_task}",
                            "details": learning_pattern,
                            "success": True
                        },
                        category="task_patterns"
                    )
                else:
                    # Learn from failure
                    await self.knowledge_manager.learn_from_experience(
                        experience={
                            "description": f"Failed to complete task: {original_task}",
                            "details": {
                                "notes": final_result.get("notes", []),
                                "steps_completed": final_result.get("steps_completed", 0),
                                "steps_total": final_result.get("steps_total", len(subtask_results))
                            },
                            "success": False
                        },
                        category="error_patterns"
                    )
                
                # Extract any insights from the synthesis result
                await self.knowledge_manager.extract_insights(synthesis_result)
                
                # Process any pending insights
                await self.knowledge_manager.process_pending_insights()
                
                return final_result
            
        except Exception as e:
            logger.error(f"Error extracting synthesis result: {e}")
        
        # Fallback if JSON extraction fails
        return {
            "success": False,
            "summary": "Failed to synthesize results properly",
            "steps_completed": len(subtask_results),
            "steps_total": len(subtask_results),
            "notes": ["Error in result synthesis"]
        }
    
    def _categorize_task(self, task_description: str) -> str:
        """Categorize a task based on its description"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["search", "find", "look for", "locate"]):
            return "search"
        elif any(word in task_lower for word in ["open", "launch", "start", "run"]):
            return "application_launch"
        elif any(word in task_lower for word in ["click", "press", "select", "choose"]):
            return "ui_interaction"
        elif any(word in task_lower for word in ["type", "enter", "input", "write"]):
            return "text_input"
        elif any(word in task_lower for word in ["download", "save", "copy", "file"]):
            return "file_operation"
        elif any(word in task_lower for word in ["research", "learn about", "information on"]):
            return "research"
        else:
            return "general"
    
    async def _log_reasoning(self, context: str, prompt: str, response: str = None) -> None:
        """Log reasoning steps for debugging and improvement"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "prompt": prompt,
            "response": response[:1000] + ("..." if response and len(response) > 1000 else "") if response else "Pending"
        }
        
        self.reasoning_log.append(log_entry)
        
        # This could write to a file for persistence, but we'll just keep in memory for now
        
        return log_entry
