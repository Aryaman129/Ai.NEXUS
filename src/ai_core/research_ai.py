"""
Research AI for NEXUS
Specialized AI for web searches, knowledge retrieval, and content analysis
"""
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import NEXUS modules
from .shared_memory import SharedMemory
from ..ai_services.research_service import ResearchService

logger = logging.getLogger(__name__)

class ResearchAI:
    """
    Research AI specialist that handles web searches, knowledge retrieval,
    and content analysis to provide information for NEXUS tasks.
    """
    
    def __init__(self, ollama_client, memory: SharedMemory, tool_registry):
        """Initialize the Research AI specialist"""
        self.ollama = ollama_client
        self.memory = memory
        self.tool_registry = tool_registry
        self.name = "research"
        self.reasoning_log = []
        
        # Initialize research service
        self.research_service = ResearchService()
        
        # Load research-related tools from registry
        self.research_tools = {}
        for tool_name, tool_info in self.tool_registry.get_all_tools().items():
            categories = tool_info.get("categories", [])
            research_categories = ["web_search", "information_retrieval", "research", "web_content"]
            
            if any(cat in research_categories for cat in categories):
                self.research_tools[tool_name] = tool_info
        
        logger.info(f"NEXUS Research AI initialized with {len(self.research_tools)} research tools")
    
    async def execute_subtask(self, subtask: Dict[str, Any], 
                            dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a research-related subtask
        
        Args:
            subtask: The subtask definition from the Coordinator
            dependency_results: Results from dependent subtasks
            
        Returns:
            Subtask execution results
        """
        subtask_id = subtask.get("id", "unknown")
        goal = subtask.get("goal", "")
        
        logger.info(f"Research AI executing subtask {subtask_id}: {goal}")
        
        try:
            # Log receipt of this subtask
            self.memory.log_specialist_communication(
                from_ai="coordinator",
                to_ai=self.name,
                message={"action": "received_subtask", "subtask_id": subtask_id}
            )
            
            # Formulate research strategy
            research_plan = await self._formulate_research_strategy(subtask, dependency_results)
            
            # Execute research queries
            results = await self._execute_research_queries(research_plan)
            
            # Analyze and synthesize information
            findings = await self._synthesize_information(results, subtask)
            
            # Return a structured result
            success = findings.get("success", False)
            final_result = {
                "status": "completed",
                "success": success,
                "findings": findings,
                "subtask_id": subtask_id,
                "sources": findings.get("sources", []),
                "information": findings.get("information", "No relevant information found"),
                "message": f"Research {'successful' if success else 'failed'} for subtask {subtask_id}"
            }
            
            # Log completion
            self.memory.log_specialist_communication(
                from_ai=self.name,
                to_ai="coordinator",
                message={"action": "completed_subtask", "subtask_id": subtask_id, "success": success}
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in Research AI executing subtask {subtask_id}: {e}")
            
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
    
    async def _formulate_research_strategy(self, subtask: Dict[str, Any], 
                                        dependency_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Formulate a research strategy based on the subtask"""
        goal = subtask.get("goal", "")
        
        # Get context from dependency results
        context = ""
        if dependency_results:
            context = "Information from dependent tasks:\n"
            for dep_id, dep_result in dependency_results.items():
                if "findings" in dep_result:
                    context += f"\n- From {dep_id}: {json.dumps(dep_result['findings'])[:500]}"
                else:
                    context += f"\n- From {dep_id}: {json.dumps(dep_result)[:500]}"
        
        # Create a prompt for the AI to determine the best research approach
        prompt = f"""
        I need to create a research strategy for this task:
        
        GOAL: {goal}
        
        {context}
        
        I have the following research tools available:
        {', '.join(self.research_tools.keys())}
        
        For this task, determine:
        1. What specific information I need to find
        2. The best search queries to use
        3. Whether I need detailed content extraction from specific sources
        4. What knowledge would help accomplish the goal
        
        Return your research strategy as JSON with these fields:
        - search_queries: array of search queries to execute
        - information_needed: specific information to look for
        - extract_content: boolean, whether to extract detailed content from results
        - focus_areas: specific topics or technical areas to focus on
        """
        
        # Log this reasoning step
        await self._log_reasoning("research_planning", prompt)
        
        # Get AI response for research planning
        plan_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Research AI specialist that plans information retrieval."
        )
        
        # Extract JSON from response
        plan = self._extract_json(plan_result)
        if not plan:
            logger.warning("Failed to extract research strategy from AI response, using fallback strategy")
            # Provide a fallback simple strategy
            plan = {
                "search_queries": [goal],
                "information_needed": f"Information about {goal}",
                "extract_content": True,
                "focus_areas": ["general information"]
            }
        
        logger.info(f"Created research strategy with {len(plan.get('search_queries', []))} queries")
        return plan
    
    async def _execute_research_queries(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research queries based on the strategy"""
        queries = plan.get("search_queries", [])
        extract_content = plan.get("extract_content", False)
        results = {}
        
        # If no queries specified, create a default one from the focus areas
        if not queries:
            focus_areas = plan.get("focus_areas", ["general information"])
            queries = [" ".join(focus_areas)]
        
        # Execute each query
        for i, query in enumerate(queries):
            query_id = f"query_{i+1}"
            
            try:
                # Execute the web search
                logger.info(f"Executing research query: {query}")
                search_result = await self.research_service.search(
                    query=query,
                    max_results=5,
                    extract_content=extract_content
                )
                
                # Store the result
                results[query_id] = search_result
                
                # Update tool statistics in memory
                self.memory.update_tool_stats(
                    "web_search", 
                    success="error" not in search_result
                )
                
            except Exception as e:
                logger.error(f"Error executing research query {query}: {e}")
                results[query_id] = {"error": str(e)}
        
        return results
    
    async def _synthesize_information(self, research_results: Dict[str, Any], 
                                   subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize findings from all research queries into useful information"""
        goal = subtask.get("goal", "")
        
        # Prepare a summary of the research results
        result_summary = ""
        
        for query_id, result in research_results.items():
            if "content" in result:
                # Truncate to avoid extremely long prompts
                content = result["content"]
                if len(content) > 1500:
                    content = content[:1500] + "...[truncated]"
                    
                result_summary += f"\n\nResults from {query_id}:\n{content}"
        
        # Create a prompt for the AI to synthesize findings
        prompt = f"""
        I need to synthesize findings from multiple research queries for this task:
        
        GOAL: {goal}
        
        {result_summary}
        
        Please analyze these results and:
        1. Extract the most relevant information for this task
        2. Organize it in a clear, concise manner
        3. Identify key facts and insights
        4. Determine if we found what we were looking for
        
        Return your synthesis as JSON with these fields:
        - success: boolean indicating if we found useful information
        - information: synthesized information relevant to the task
        - key_facts: array of key facts extracted from the research
        - sources: array of sources (URLs) that provided useful information
        - confidence: overall confidence in the information (0-1)
        """
        
        # Log this reasoning step
        await self._log_reasoning("information_synthesis", prompt)
        
        # Get AI response for synthesis
        synthesis_result = await self.ollama.generate_text(
            prompt,
            system_prompt="You are the NEXUS Research AI specialist that synthesizes information."
        )
        
        # Extract JSON from response
        findings = self._extract_json(synthesis_result)
        if not findings:
            logger.warning("Failed to extract findings from AI response, using basic findings")
            
            # Create basic findings from the raw results
            sources = []
            information = []
            
            # Extract information and sources from each result
            for query_id, result in research_results.items():
                if "content" in result:
                    information.append(result["content"][:500] + "...")
                if "results" in result:
                    for item in result["results"]:
                        if "url" in item and item["url"] not in sources:
                            sources.append(item["url"])
            
            findings = {
                "success": len(information) > 0,
                "information": "\n\n".join(information),
                "key_facts": ["Information synthesized from raw results"],
                "sources": sources[:5],  # Limit to first 5 sources
                "confidence": 0.5  # Medium confidence as this is a fallback
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
