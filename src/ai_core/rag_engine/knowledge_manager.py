"""
Knowledge Manager for NEXUS RAG Engine
Provides a comprehensive system for storing, retrieving, and learning from information
"""
import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .vector_storage import VectorStorage

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Knowledge Manager for NEXUS that provides RAG capabilities.
    
    This component:
    1. Stores knowledge in vectorized form for semantic retrieval
    2. Enhances LLM reasoning with relevant retrieved context
    3. Dynamically learns from new experiences and observations
    4. Prioritizes information based on recency and relevance
    """
    
    def __init__(self, 
                 ollama_client=None, 
                 vector_storage: Optional[VectorStorage] = None,
                 knowledge_base_dir: str = "memory/knowledge"):
        """
        Initialize the knowledge manager
        
        Args:
            ollama_client: LLM client for text generation
            vector_storage: Optional pre-configured vector storage
            knowledge_base_dir: Directory to store knowledge base
        """
        self.ollama = ollama_client
        self.knowledge_dir = Path(knowledge_base_dir)
        self.knowledge_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up vector storage if not provided
        if vector_storage is None:
            self.vector_storage = VectorStorage(
                storage_dir=str(self.knowledge_dir / "vector_db"),
                backend="auto",
                collection_name="nexus_knowledge"
            )
        else:
            self.vector_storage = vector_storage
            
        # Knowledge categories
        self.knowledge_categories = {
            "task_patterns": "Successful task execution patterns",
            "tool_usage": "Information about tools and their usage",
            "user_preferences": "User preferences and patterns",
            "system_knowledge": "NEXUS system capabilities and limits",
            "external_knowledge": "Knowledge from web search and external sources",
            "reasoning_patterns": "Successful reasoning approaches",
            "error_patterns": "Common errors and how to avoid them"
        }
        
        # Session-specific context (cleared between runs)
        self.session_context = {
            "recent_retrievals": [],
            "pending_insights": [],
            "created_at": datetime.now().isoformat()
        }
        
        logger.info("NEXUS Knowledge Manager initialized")
    
    async def add_knowledge(self, 
                          text: str, 
                          metadata: Dict[str, Any], 
                          category: str) -> str:
        """
        Add knowledge to the system
        
        Args:
            text: The text content to add
            metadata: Additional metadata about the content
            category: Knowledge category (task_patterns, tool_usage, etc.)
            
        Returns:
            ID of the added knowledge
        """
        if not text or not category:
            logger.warning("Attempted to add empty knowledge")
            return ""
        
        # Ensure the category is valid
        if category not in self.knowledge_categories:
            logger.warning(f"Unknown knowledge category: {category}, defaulting to 'system_knowledge'")
            category = "system_knowledge"
            
        # Prepare metadata
        full_metadata = {
            "category": category,
            "added_at": datetime.now().isoformat(),
            "source": metadata.get("source", "nexus_internal"),
            **metadata
        }
        
        # Add text to vector storage
        ids = self.vector_storage.add_texts(
            texts=[text],
            metadatas=[full_metadata]
        )
        
        # Log the addition
        logger.info(f"Added knowledge to {category}: {text[:50]}...")
        
        return ids[0] if ids else ""
    
    async def retrieve_relevant_knowledge(self, 
                                         query: str, 
                                         category: Optional[str] = None,
                                         max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge relevant to a query
        
        Args:
            query: The query to find relevant knowledge for
            category: Optional category to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        if not query:
            return []
            
        # Create filter if category specified
        filter_dict = {"category": category} if category else None
        
        # Perform semantic search
        results = self.vector_storage.similarity_search(
            query=query,
            k=max_results,
            filter_dict=filter_dict
        )
        
        # Update session context
        self.session_context["recent_retrievals"] = [
            {
                "query": query,
                "results": [r["id"] for r in results],
                "timestamp": datetime.now().isoformat()
            }
        ] + self.session_context["recent_retrievals"][:4]  # Keep last 5
        
        return results
    
    async def enhance_with_knowledge(self, 
                                   original_prompt: str,
                                   task_description: str = "",
                                   relevant_categories: List[str] = None) -> str:
        """
        Enhance a prompt with relevant knowledge
        
        Args:
            original_prompt: The original prompt to enhance
            task_description: Description of the current task for better context retrieval
            relevant_categories: Categories of knowledge to include
            
        Returns:
            Enhanced prompt with relevant knowledge
        """
        if not original_prompt:
            return original_prompt
            
        # Default to all categories if none specified
        if not relevant_categories:
            relevant_categories = list(self.knowledge_categories.keys())
            
        # Combine the task description and prompt for better retrieval
        retrieval_query = task_description + " " + original_prompt if task_description else original_prompt
        
        # Initialize enhanced prompt
        enhanced_prompt = original_prompt
        relevant_context = []
        
        # Retrieve relevant knowledge from each category
        for category in relevant_categories:
            results = await self.retrieve_relevant_knowledge(
                query=retrieval_query,
                category=category,
                max_results=2  # Limit per category to avoid too much context
            )
            
            if results:
                category_name = self.knowledge_categories.get(category, category)
                category_context = f"\n\nRelevant {category_name}:"
                
                for result in results:
                    score = result.get("score", 0)
                    if score > 0.6:  # Only include if reasonably relevant
                        category_context += f"\n- {result['text']}"
                
                if category_context != f"\n\nRelevant {category_name}:":
                    relevant_context.append(category_context)
        
        # Add relevant context to the prompt
        if relevant_context:
            context_section = "\n\n### Relevant Context ###" + "".join(relevant_context)
            enhanced_prompt = enhanced_prompt + context_section
        
        return enhanced_prompt
    
    async def learn_from_experience(self, 
                                  experience: Dict[str, Any], 
                                  category: str) -> str:
        """
        Learn from a new experience
        
        Args:
            experience: Dictionary containing the experience details
            category: Knowledge category for this experience
            
        Returns:
            ID of the added knowledge
        """
        if not experience:
            return ""
            
        # Extract text and metadata from experience
        if isinstance(experience, dict):
            # Extract text from the experience dictionary
            text = experience.get("description", "")
            if not text and "details" in experience:
                text = json.dumps(experience["details"])
                
            if not text:
                # Create text from the entire experience
                text = json.dumps(experience)
                
            # Use the rest as metadata
            metadata = {k: v for k, v in experience.items() if k != "description"}
            
        else:
            # Handle case where experience is a string
            text = str(experience)
            metadata = {}
        
        # Add learning timestamp
        metadata["learned_at"] = datetime.now().isoformat()
        
        # Add to knowledge base
        return await self.add_knowledge(text, metadata, category)
    
    async def extract_insights(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract insights from text that might be worth learning
        
        Args:
            text: Text to extract insights from
            
        Returns:
            List of extracted insights with category suggestions
        """
        if not text or not self.ollama:
            return []
            
        # Create prompt for insight extraction
        prompt = f"""
        Analyze this text and extract key insights that would be valuable for an autonomous AI system to learn:
        
        TEXT: {text}
        
        For each insight:
        1. Extract the core learning or pattern
        2. Identify which category it belongs to: task_patterns, tool_usage, user_preferences, system_knowledge, reasoning_patterns, error_patterns
        3. Explain why this is valuable to learn
        
        Return your analysis as a JSON array, where each object follows this format:
        {{
            "insight": "The extracted insight in a clear, concise format",
            "category": "one_of_the_categories_above",
            "value": "Brief explanation of why this is valuable to learn",
            "confidence": a number between 0 and 1 indicating how confident you are this is an important insight
        }}
        
        Only extract insights with confidence 0.7 or higher.
        """
        
        # Get insights from LLM
        try:
            insight_extraction = await self.ollama.generate_text(
                prompt,
                system_prompt="You are an insight extraction system for NEXUS Intelligence."
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', insight_extraction)
            if json_match:
                insights = json.loads(json_match.group(0))
                
                # Filter by confidence
                insights = [i for i in insights if i.get("confidence", 0) >= 0.7]
                
                # Add to pending insights
                self.session_context["pending_insights"].extend(insights)
                
                return insights
                
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            
        return []
    
    async def process_pending_insights(self) -> int:
        """
        Process and learn from pending insights
        
        Returns:
            Number of insights processed
        """
        if not self.session_context["pending_insights"]:
            return 0
            
        processed_count = 0
        
        # Process each pending insight
        for insight in self.session_context["pending_insights"]:
            category = insight.get("category", "system_knowledge")
            
            # Format the insight text
            insight_text = f"{insight.get('insight', '')} - {insight.get('value', '')}"
            
            # Add to knowledge base
            await self.add_knowledge(
                text=insight_text,
                metadata={
                    "confidence": insight.get("confidence", 0.7),
                    "source": "insight_extraction",
                    "original_text": insight.get("original_text", "")
                },
                category=category
            )
            
            processed_count += 1
            
        # Clear pending insights
        self.session_context["pending_insights"] = []
        
        return processed_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge storage"""
        return self.vector_storage.get_stats()
    
    async def summarize_category(self, category: str) -> str:
        """
        Summarize knowledge in a specific category
        
        Args:
            category: The category to summarize
            
        Returns:
            Summary of the knowledge in the category
        """
        if category not in self.knowledge_categories:
            return f"Unknown category: {category}"
            
        # Retrieve all items in the category
        filter_dict = {"category": category}
        
        # Use a generic query to get items in this category
        results = self.vector_storage.similarity_search(
            query=self.knowledge_categories[category],
            k=20,  # Get up to 20 items
            filter_dict=filter_dict
        )
        
        if not results:
            return f"No knowledge found in category: {category}"
            
        # Create a summary using the LLM if available
        if self.ollama:
            texts = [r["text"] for r in results]
            
            prompt = f"""
            Please summarize the key points from this knowledge collection on {self.knowledge_categories[category]}:
            
            {texts}
            
            Provide a concise summary that captures the most important patterns and insights. Focus on actionable knowledge.
            """
            
            try:
                summary = await self.ollama.generate_text(
                    prompt,
                    system_prompt="You are a knowledge summarization system for NEXUS Intelligence."
                )
                return summary
                
            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                
        # Fallback: simple concatenation of texts
        return "\n\n".join([f"- {r['text']}" for r in results])
        
    async def analyze_task_success(self, 
                                 task_description: str,
                                 execution_log: List[Dict[str, Any]],
                                 success: bool) -> Dict[str, Any]:
        """
        Analyze task execution to extract learnings
        
        Args:
            task_description: Description of the task
            execution_log: Log of the task execution
            success: Whether the task was successful
            
        Returns:
            Analysis with extracted learnings
        """
        if not task_description or not execution_log:
            return {"success": False, "message": "Insufficient data for analysis"}
            
        # If LLM is not available, just store the basic outcome
        if not self.ollama:
            category = "task_patterns" if success else "error_patterns"
            outcome = "successful" if success else "failed"
            
            await self.add_knowledge(
                text=f"Task '{task_description}' {outcome}",
                metadata={
                    "success": success,
                    "task": task_description
                },
                category=category
            )
            
            return {
                "success": True,
                "message": f"Basic {outcome} task record stored",
                "learnings": []
            }
            
        # Create prompt for task analysis
        execution_summary = json.dumps(execution_log[:10])  # Limit to first 10 items
        
        prompt = f"""
        Analyze this {'successful' if success else 'failed'} task execution and extract key learnings:
        
        TASK: {task_description}
        
        EXECUTION LOG:
        {execution_summary}
        
        Please identify:
        1. What approach was used for this task
        2. Key factors that {'led to success' if success else 'caused failure'}
        3. Patterns worth remembering for future tasks
        4. {'What could be improved' if success else 'How to avoid this failure in the future'}
        
        Return your analysis as JSON with these fields:
        - approach: The approach used for the task
        - key_factors: Array of key factors
        - patterns: Array of patterns worth remembering
        - improvements: Array of possible improvements
        - category: "task_patterns" or "error_patterns" or "reasoning_patterns"
        """
        
        try:
            analysis_result = await self.ollama.generate_text(
                prompt,
                system_prompt="You are a task analysis system for NEXUS Intelligence."
            )
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis_result)
            if json_match:
                analysis = json.loads(json_match.group(0))
                
                # Store the analysis
                category = analysis.get("category", "task_patterns" if success else "error_patterns")
                
                # Store the overall approach
                approach_text = f"For task '{task_description}': {analysis.get('approach', '')}"
                await self.add_knowledge(
                    text=approach_text,
                    metadata={
                        "success": success,
                        "task": task_description,
                        "type": "approach"
                    },
                    category=category
                )
                
                # Store each pattern
                learnings = []
                for pattern in analysis.get("patterns", []):
                    pattern_id = await self.add_knowledge(
                        text=pattern,
                        metadata={
                            "success": success,
                            "task": task_description,
                            "type": "pattern"
                        },
                        category=category
                    )
                    learnings.append({"id": pattern_id, "text": pattern})
                
                return {
                    "success": True,
                    "message": "Task analysis completed and learnings stored",
                    "learnings": learnings,
                    "analysis": analysis
                }
                
        except Exception as e:
            logger.error(f"Error analyzing task: {e}")
            
        # Fallback: store basic outcome
        category = "task_patterns" if success else "error_patterns"
        outcome = "successful" if success else "failed"
        
        await self.add_knowledge(
            text=f"Task '{task_description}' {outcome}",
            metadata={
                "success": success,
                "task": task_description
            },
            category=category
        )
        
        return {
            "success": True,
            "message": f"Basic {outcome} task record stored (analysis failed)",
            "learnings": []
        }
