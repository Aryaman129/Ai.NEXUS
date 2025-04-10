"""
NEXUS AI Core
Multi-specialist AI architecture with coordinated reasoning
"""

from .shared_memory import SharedMemory
from .coordinator_ai import CoordinatorAI
from .vision_ai import VisionAI
from .research_ai import ResearchAI
from .automation_ai import AutomationAI
from .nexus_ai_orchestrator import NexusAIOrchestrator

# Import advanced components if available
try:
    from .rag_engine import KnowledgeManager, VectorStorage
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from .ml_models import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
# Import screen analysis components
try:
    from .screen_analysis import VisualMemorySystem, DETECTORS_AVAILABLE
    SCREEN_ANALYSIS_AVAILABLE = True
except ImportError:
    SCREEN_ANALYSIS_AVAILABLE = False
    DETECTORS_AVAILABLE = False

__all__ = [
    'SharedMemory',
    'CoordinatorAI',
    'VisionAI', 
    'ResearchAI',
    'AutomationAI',
    'NexusAIOrchestrator'
]

if RAG_AVAILABLE:
    __all__.extend(['KnowledgeManager', 'VectorStorage'])

if YOLO_AVAILABLE:
    __all__.append('YOLODetector')
    
if SCREEN_ANALYSIS_AVAILABLE:
    __all__.append('VisualMemorySystem')
