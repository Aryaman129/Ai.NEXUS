# NEXUS: Autonomous AI Orchestration System

NEXUS is an advanced AI orchestration system that autonomously selects, combines, and orchestrates tools based on adaptive learning rather than rigid rules.

## Core Philosophy

**"The AI should learn and adapt instead of following rigid rules."**

NEXUS embodies this philosophy by implementing a flexible, experience-driven architecture that adjusts to changing circumstances, available tools, and user preferences.

## Core Architecture

The fundamental architecture of NEXUS is based on AI-orchestrated tool integration where:

1. **AI-Driven Tool Selection**: The AI autonomously selects, combines, and orchestrates tools without hard-coded paths
2. **Dynamic Tool Combinations**: AI creates new tool combinations dynamically based on task requirements
3. **Tool Registry System**: Tools register with a central registry that the AI can discover and utilize
4. **Web Knowledge Integration**: Web search capabilities allow the AI to learn new approaches and gather information
5. **Continuous Learning**: Success/failure tracking automatically optimizes tool selection over time
6. **Adaptive Enhancement**: The AI can suggest new libraries or tools to enhance capabilities
7. **Interactive Clarification**: Interactive learning allows the AI to ask clarifying questions when uncertain
8. **Memory & Personalization**: Memory systems store successful patterns and user preferences for personalization

## Key Features

- **Adaptive LLM Selection**: Dynamically selects the most appropriate LLM based on task type, past performance, and availability
- **RAG Pipeline with Adaptive Search**: Retrieval-Augmented Generation that combines vector search with web search
- **Shared Memory System**: Learns from past experiences to improve future task performance
- **Semantic Expansion for Task Matching**: Matches new tasks to prior experiences using semantic relationships
- **Dynamic Integration Manager**: Discovers and utilizes available tools at runtime
- **Tiered Vision Analysis**: Multi-tiered approach for analyzing screenshots, combining AI vision, OCR, and computer vision
- **Web Integration**: DuckDuckGo web search capabilities without external API keys
- **Progressive Learning**: The system improves its performance over time through user feedback and learning from past interactions
- **Adaptive Screen Automation**: Self-learning automation system that adapts to UI changes and builds confidence through experience
- **Visual Memory System**: Stores and recognizes UI patterns across applications for faster, more confident interactions
- **Clarification Engine**: Intelligently asks questions when uncertain and adapts confidence thresholds based on responses
- **Floating Assistant UI**: Always-on-top interface that allows NEXUS to interact with the user while controlling other applications
- **Multi-Application Orchestration**: Monitors and controls multiple applications simultaneously with cross-app workflows

## Integrations

NEXUS adapts to available AI resources through its unified integration frameworks:

### Unified LLM Integration Framework
- **Ollama**: Local model inference with models like deepseek-coder, llava, and dolphin-phi
- **Groq**: Fast cloud-based inference with models like Mixtral and Llama
- **Hugging Face**: Access to thousands of open models for various AI tasks
- **Together AI**: Enhanced LLM capabilities with various models
- **Mistral AI**: Specialized model access
- **OpenRouter**: Multi-model access through a single API
- **Gemini**: Text and vision capabilities

### Unified Vision System
- **Gemini Vision API**: Primary vision capability when available
- **OpenCV**: Local computer vision processing
- **EasyOCR**: Text extraction from images
- **Multiple UI Detection Backends**: AutoGluon, YOLO, HuggingFace, OpenCV

### Other Integrations
- **DuckDuckGo Search**: Web search capabilities without API keys

## Getting Started

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the integration tests:
   ```
   python test_nexus_integration.py
   ```

3. Run the demo:
   ```
   python src/run_nexus.py
   ```

## Project Structure

- `src/` - Core NEXUS source code
  - `ai_core/` - Core AI components
    - `rag_engine/` - Retrieval-augmented generation system
    - `screen_analysis/` - Screen analysis and UI detection
      - `ui_detection/` - Unified UI detection framework
    - `shared_memory/` - Memory and learning systems
    - `coordinator.py` - Task coordination and decomposition
  - `integrations/` - Integration modules for external services
    - `llm_integration/` - Unified LLM integration framework
  - `modules/` - Specific functional modules
  - `nexus_core.py` - Main NEXUS implementation
- `tests/` - Test suite for validating NEXUS components
- `templates/` - Template files for computer vision
- `docs/` - Documentation

## Dependencies

- Python 3.8+
- PyAutoGUI & PyGetWindow - For screen automation
- OpenCV & NumPy - For image processing and UI detection
- PyTorch - For GPU-accelerated model inference
- Tkinter - For floating assistant UI
- Scikit-image - For visual patterns analysis
- See `requirements.txt` for complete list of dependencies

## Planned Enhancements

- **Cross-Application Learning**: Transfer knowledge between similar applications
- **Neural Reinforcement Learning**: Accelerate adaptation using reinforcement learning
- **Multimodal Understanding**: Better understand screen content semantically using vision-language models
- **Community Pattern Sharing**: Securely share successful UI interaction patterns between NEXUS instances
- **Voice Interface**: Add voice commands and responses for hands-free operation
- **Multi-Application Orchestration**: Monitor and control multiple applications simultaneously with cross-app workflows
