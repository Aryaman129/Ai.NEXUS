"""
NEXUS Unified LLM Integration Framework

This module provides a unified framework for LLM integration with multiple providers.
It consolidates previously scattered LLM integration implementations into a single,
modular system that embodies the NEXUS philosophy of adaptation over rigid rules.
"""

from .llm_interface import LLMInterface
from .llm_registry import LLMRegistry
from .adaptive_llm_manager import AdaptiveLLMManager

# Import available providers
try:
    from .ollama_provider import OllamaProvider
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from .groq_provider import GroqProvider
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from .huggingface_provider import HuggingFaceProvider
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from .mistral_provider import MistralProvider
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    from .openrouter_provider import OpenRouterProvider
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    from .together_provider import TogetherProvider
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

try:
    from .gemini_provider import GeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Version tracking
__version__ = "0.2.0"
