"""
Shared Memory & Context Store for NEXUS
Maintains context between specialist AIs to prevent knowledge loss
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class SharedMemory:
    """
    Shared memory system that maintains context between specialist AIs
    and provides persistent storage of learning experiences.
    """
    
    def __init__(self, memory_file: str = "nexus_memory.json"):
        """Initialize the shared memory system"""
        self.memory_file = memory_file
        self.memory_dir = Path("memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Main memory structure
        self.memory = {
            "tasks": {},           # Task history with outcomes
            "user_preferences": {},  # User preferences and patterns
            "tool_stats": {},      # Tool usage statistics
            "learned_patterns": {},  # Learned patterns for tasks
            "error_patterns": {},  # Common error patterns to avoid
        }
        
        # Temporary session context (cleared between runs)
        self.context = {
            "current_task": {},
            "task_decomposition": [],
            "intermediate_results": {},
            "active_specialists": [],
            "communication_history": [],
            "context_created_at": datetime.now().isoformat()
        }
        
        # Load existing memory if available
        self._load_memory()
        
        logger.info("NEXUS Shared Memory initialized")
    
    def _load_memory(self) -> bool:
        """Load memory from disk if available"""
        memory_path = self.memory_dir / self.memory_file
        
        if memory_path.exists():
            try:
                with open(memory_path, 'r') as f:
                    stored_memory = json.load(f)
                
                # Update memory with stored values
                for key, value in stored_memory.items():
                    if key in self.memory:
                        self.memory[key] = value
                
                logger.info(f"Loaded memory from {memory_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                return False
        else:
            logger.info("No existing memory file found, using empty memory")
            return False
    
    def save_memory(self) -> bool:
        """Save memory to disk"""
        memory_path = self.memory_dir / self.memory_file
        
        try:
            with open(memory_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
            
            logger.info(f"Saved memory to {memory_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def create_session_context(self, task_description: str) -> Dict[str, Any]:
        """Create a new session context for a task"""
        # Reset context for new task
        self.context = {
            "current_task": {
                "description": task_description,
                "created_at": datetime.now().isoformat(),
                "status": "started"
            },
            "task_decomposition": [],
            "intermediate_results": {},
            "active_specialists": [],
            "communication_history": [],
            "context_created_at": datetime.now().isoformat()
        }
        
        # Create a snapshot file for debugging
        context_snapshot = self.memory_dir / f"context_{int(time.time())}.json"
        with open(context_snapshot, 'w') as f:
            json.dump(self.context, f, indent=2)
        
        return self.context
    
    def update_context(self, key: str, value: Any) -> bool:
        """Update a specific key in the current context"""
        if key in self.context:
            self.context[key] = value
            return True
        else:
            logger.warning(f"Unknown context key: {key}")
            return False
    
    def add_to_context(self, key: str, value: Any) -> bool:
        """Add or append to a context value"""
        if key in self.context:
            if isinstance(self.context[key], list):
                self.context[key].append(value)
            elif isinstance(self.context[key], dict):
                if isinstance(value, dict):
                    self.context[key].update(value)
                else:
                    logger.warning(f"Cannot update dict context with non-dict value")
                    return False
            else:
                # Replace the value
                self.context[key] = value
            return True
        else:
            logger.warning(f"Unknown context key: {key}")
            return False
    
    def get_context(self, key: Optional[str] = None) -> Any:
        """Get current context or a specific key"""
        if key is not None:
            return self.context.get(key)
        return self.context
    
    def log_specialist_communication(self, from_ai: str, to_ai: str, message: Dict[str, Any]) -> None:
        """Log communication between specialist AIs"""
        communication_entry = {
            "timestamp": datetime.now().isoformat(),
            "from": from_ai,
            "to": to_ai,
            "message": message
        }
        
        self.context["communication_history"].append(communication_entry)
    
    def store_task_result(self, task_result: Dict[str, Any]) -> None:
        """Store the result of a completed task in persistent memory"""
        task_description = self.context["current_task"].get("description", "unknown task")
        
        # Update the task status
        self.context["current_task"]["status"] = "completed" if task_result.get("success", False) else "failed"
        self.context["current_task"]["completed_at"] = datetime.now().isoformat()
        self.context["current_task"]["result"] = task_result
        
        # Store in persistent memory
        self.memory["tasks"][task_description] = {
            "description": task_description,
            "last_executed": datetime.now().isoformat(),
            "success": task_result.get("success", False),
            "steps_completed": task_result.get("steps_completed", 0),
            "steps_total": task_result.get("steps_total", 0),
            "tools_used": task_result.get("tools_used", []),
            "specialists_involved": self.context.get("active_specialists", []),
            "notes": task_result.get("notes", [])
        }
        
        # Save memory to disk
        self.save_memory()
    
    def update_tool_stats(self, tool_name: str, success: bool) -> None:
        """Update usage statistics for a tool"""
        if tool_name not in self.memory["tool_stats"]:
            self.memory["tool_stats"][tool_name] = {
                "uses": 0,
                "successes": 0,
                "failures": 0,
                "last_used": None
            }
        
        self.memory["tool_stats"][tool_name]["uses"] += 1
        if success:
            self.memory["tool_stats"][tool_name]["successes"] += 1
        else:
            self.memory["tool_stats"][tool_name]["failures"] += 1
        
        self.memory["tool_stats"][tool_name]["last_used"] = datetime.now().isoformat()
    
    def learn_from_success(self, pattern: Dict[str, Any]) -> None:
        """Record a successful pattern for future reference"""
        task_type = pattern.get("task_type", "general")
        
        if task_type not in self.memory["learned_patterns"]:
            self.memory["learned_patterns"][task_type] = []
        
        # Store the pattern with a timestamp
        pattern["learned_at"] = datetime.now().isoformat()
        self.memory["learned_patterns"][task_type].append(pattern)
        
        # Keep only the most recent 10 patterns per task type
        if len(self.memory["learned_patterns"][task_type]) > 10:
            self.memory["learned_patterns"][task_type] = self.memory["learned_patterns"][task_type][-10:]
        
        self.save_memory()
    
    def learn_from_error(self, error_pattern: Dict[str, Any]) -> None:
        """Record an error pattern to avoid in the future"""
        error_type = error_pattern.get("error_type", "general")
        
        if error_type not in self.memory["error_patterns"]:
            self.memory["error_patterns"][error_type] = []
        
        # Store the error pattern with a timestamp
        error_pattern["learned_at"] = datetime.now().isoformat()
        self.memory["error_patterns"][error_type].append(error_pattern)
        
        # Keep only the most recent 10 error patterns per type
        if len(self.memory["error_patterns"][error_type]) > 10:
            self.memory["error_patterns"][error_type] = self.memory["error_patterns"][error_type][-10:]
        
        self.save_memory()
    
    def update_user_preference(self, preference_name: str, preference_value: Any) -> None:
        """Update a user preference"""
        self.memory["user_preferences"][preference_name] = {
            "value": preference_value,
            "updated_at": datetime.now().isoformat()
        }
        
        self.save_memory()
    
    def get_related_experiences(self, task_description: str) -> List[Dict[str, Any]]:
        """Get past experiences related to the current task"""
        related_experiences = []
        
        # Simple keyword matching with expanded semantics
        # Could be enhanced with embedding similarity in a production system
        task_words = task_description.lower().split()
        
        # Semantic expansion - add related terms to improve matching
        semantic_expansions = {
            "neural": ["deep learning", "machine learning", "ai", "artificial intelligence", "ml", "network"],
            "network": ["neural", "deep learning", "graph", "connection"],
            "information": ["data", "search", "find", "retrieve", "query", "knowledge"],
            "find": ["search", "discover", "retrieve", "query", "get"],
            "deep learning": ["neural network", "machine learning", "ai", "artificial intelligence"],
            "search": ["find", "query", "retrieve", "lookup", "research", "information"],
            "algorithm": ["model", "method", "technique", "approach", "solution", "system"],
            "generate": ["create", "produce", "make", "develop", "write", "compose"],
            "text": ["content", "writing", "document", "passage", "article"]
        }
        
        # Expand task words with related terms
        expanded_task_words = task_words.copy()
        for word in task_words:
            # Check if any keys in semantic_expansions are contained in the word
            for key, expansions in semantic_expansions.items():
                if key in word:  # Partial match like "neural" in "neurons"
                    expanded_task_words.extend(expansions)
                    break
        
        # Remove duplicates while preserving order
        expanded_task_words = list(dict.fromkeys(expanded_task_words))
        
        logger.info(f"Expanded task terms: {expanded_task_words}")
        
        # Check patterns in both success and error collections
        # First check success patterns
        for task_type, patterns in self.memory["learned_patterns"].items():
            for pattern in patterns:
                pattern_desc = pattern.get("description", "")
                pattern_words = pattern_desc.lower().split()
                
                # Check if any words match between descriptions (using expanded terms)
                if pattern_desc and any(word in pattern_desc.lower() for word in expanded_task_words):
                    match_score = sum(1 for word in expanded_task_words if word in pattern_desc.lower())
                    related_experiences.append({
                        "source": "success_pattern",
                        "task_type": task_type,
                        "description": pattern_desc,
                        "details": pattern,
                        "outcome": "completed_successfully",
                        "approach": pattern.get("approach", {}),
                        "match_score": match_score
                    })
        
        # Then check error patterns
        for error_type, patterns in self.memory["error_patterns"].items():
            for pattern in patterns:
                pattern_desc = pattern.get("description", "")
                
                # Check if any words match between descriptions (using expanded terms)
                if pattern_desc and any(word in pattern_desc.lower() for word in expanded_task_words):
                    match_score = sum(1 for word in expanded_task_words if word in pattern_desc.lower())
                    related_experiences.append({
                        "source": "error_pattern",
                        "error_type": error_type,
                        "description": pattern_desc,
                        "details": pattern,
                        "outcome": "failed",
                        "approach": pattern.get("approach", {}),
                        "match_score": match_score
                    })
        
        # Also check completed tasks
        for task_key, task_data in self.memory["tasks"].items():
            if any(word in task_key.lower() for word in expanded_task_words):
                match_score = sum(1 for word in expanded_task_words if word in task_key.lower())
                task_data["match_score"] = match_score
                related_experiences.append(task_data)
        
        # Sort by relevance - prioritize:
        # 1. Success over failure
        # 2. Higher match score (more matching terms)
        related_experiences.sort(key=lambda x: (-1 if x.get("outcome") == "completed_successfully" else 0, 
                                               -x.get("match_score", 0)))
        
        logger.info(f"Found {len(related_experiences)} experiences related to '{task_description}'")
        
        return related_experiences
    
    def get_tool_recommendations(self, task_type: str) -> List[str]:
        """Get tool recommendations based on past successful tasks"""
        successful_tools = []
        
        # Find tools that were successful for similar tasks
        for task_key, task_data in self.memory["tasks"].items():
            if task_type.lower() in task_key.lower() and task_data.get("success", False):
                successful_tools.extend(task_data.get("tools_used", []))
        
        # Count occurrences and sort by frequency
        tool_counts = {}
        for tool in successful_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        recommended_tools = sorted(tool_counts.keys(), key=lambda t: tool_counts[t], reverse=True)
        return recommended_tools
