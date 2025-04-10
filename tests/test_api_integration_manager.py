"""
Tests for the API Integration Manager

This module tests the API Integration Manager's capability to coordinate
multiple API services, manage rate limits, and implement knowledge distillation.
"""
import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from src.ai_core.api_integration_manager import APIIntegrationManager, RateLimitManager, APIKeyManager

# Mock integration class for testing
class MockIntegration:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.available = True
        
    async def initialize(self):
        return self.available
        
    async def get_capabilities(self):
        return ["text_generation", "code_generation"]
        
    async def execute(self, capability, **kwargs):
        if capability == "text_generation":
            return {
                "text": "This is a test response from mock integration",
                "model": "mock-model"
            }
        elif capability == "code_generation":
            return {
                "text": "def hello_world():\n    print('Hello, World!')",
                "model": "mock-code-model"
            }
        else:
            return {"error": "Unsupported capability"}


@pytest.fixture
def mock_shared_memory():
    memory = MagicMock()
    memory.add_experience = MagicMock()
    return memory


@pytest.fixture
def mock_vector_storage():
    storage = MagicMock()
    storage.add_texts = MagicMock()
    return storage


@pytest.fixture
def api_manager(mock_shared_memory, mock_vector_storage):
    return APIIntegrationManager(
        shared_memory=mock_shared_memory,
        vector_storage=mock_vector_storage
    )


@pytest.mark.asyncio
async def test_initialize_providers():
    """Test initialization of providers"""
    # Mock all integrations
    with patch('src.ai_core.api_integration_manager.OpenRouterIntegration', MockIntegration), \
         patch('src.ai_core.api_integration_manager.MistralAIIntegration', MockIntegration), \
         patch('src.ai_core.api_integration_manager.HuggingFaceIntegration', MockIntegration):
        
        manager = APIIntegrationManager()
        providers = await manager.initialize_providers()
        
        # Check that providers were registered
        assert len(providers) == 3
        assert "openrouter" in providers
        assert "mistral" in providers
        assert "huggingface" in providers
        
        # Check that capabilities were registered
        assert "text_generation" in manager.capabilities
        assert "code_generation" in manager.capabilities
        assert len(manager.capabilities["text_generation"]) == 3


@pytest.mark.asyncio
async def test_execute_capability(api_manager):
    """Test executing a capability"""
    # Add a mock provider
    api_manager.providers = {"mock": MockIntegration()}
    api_manager.capabilities = {"text_generation": ["mock"]}
    
    # Execute the capability
    result = await api_manager._execute_with_provider(
        provider="mock",
        capability="text_generation",
        prompt="Hello"
    )
    
    # Check result
    assert "error" not in result
    assert result["text"] == "This is a test response from mock integration"
    assert result["provider"] == "mock"


@pytest.mark.asyncio
async def test_provider_selection(api_manager):
    """Test selection of the best provider"""
    # Set up providers and capabilities
    api_manager.providers = {
        "provider1": MockIntegration(),
        "provider2": MockIntegration(),
        "provider3": MockIntegration()
    }
    api_manager.capabilities = {
        "text_generation": ["provider1", "provider2", "provider3"]
    }
    
    # Add performance metrics
    api_manager.performance_metrics = {
        "provider1:text_generation": {
            "success_rate": 0.9,
            "avg_time": 0.5,
            "tasks": {"general": {"success_rate": 0.95}}
        },
        "provider2:text_generation": {
            "success_rate": 0.8,
            "avg_time": 0.3,
            "tasks": {"general": {"success_rate": 0.85}}
        },
        "provider3:text_generation": {
            "success_rate": 0.7,
            "avg_time": 0.2,
            "tasks": {"general": {"success_rate": 0.75}}
        }
    }
    
    # Select provider for general task
    provider = await api_manager._select_best_provider("text_generation", "general")
    
    # Provider1 should be selected (highest success rate)
    assert provider == "provider1"
    
    # Now mark provider1 as rate-limited
    api_manager.rate_limit_manager.rate_limits["provider1"]["rate_limited"] = True
    
    # Select again
    provider = await api_manager._select_best_provider("text_generation", "general")
    
    # Provider2 should be selected now
    assert provider == "provider2"


@pytest.mark.asyncio
async def test_knowledge_distillation(api_manager, mock_vector_storage, mock_shared_memory):
    """Test knowledge distillation from API responses"""
    # Add a successful result to the learning buffer
    await api_manager._record_for_knowledge_distillation(
        capability="text_generation",
        prompt="What is NEXUS?",
        result={
            "text": "NEXUS is an AI orchestration system.",
            "model": "test-model"
        },
        provider="test-provider",
        task_type="general"
    )
    
    # Manually process the buffer
    assert len(api_manager.learning_buffer) == 1
    await api_manager._process_learning_buffer()
    
    # Check that vector storage and shared memory were updated
    assert mock_vector_storage.add_texts.called
    assert mock_shared_memory.add_experience.called
    
    # Buffer should be cleared
    assert len(api_manager.learning_buffer) == 0


@pytest.mark.asyncio
async def test_rate_limit_manager():
    """Test rate limit management"""
    rate_limit_manager = RateLimitManager()
    
    # Check initial state
    assert rate_limit_manager.is_available("openrouter")
    
    # Mark as rate limited
    rate_limit_manager.mark_rate_limited("openrouter")
    
    # Should be unavailable now
    assert not rate_limit_manager.is_available("openrouter")
    
    # Check wait behavior
    start_time = asyncio.get_event_loop().time()
    await rate_limit_manager.wait_if_needed("mistral")
    # Should wait for minimum interval (1/0.5 = 2 seconds)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # Should be close to zero for first call
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_api_key_manager():
    """Test API key management"""
    # Mock environment variables
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-openrouter-key",
        "MISTRAL_API_KEY": "test-mistral-key"
    }):
        key_manager = APIKeyManager()
        
        # Check key retrieval
        assert key_manager.get_next_available_key("openrouter") == "test-openrouter-key"
        assert key_manager.get_next_available_key("mistral") == "test-mistral-key"
        
        # Mark a key as rate limited
        key_manager.mark_key_rate_limited("openrouter", "test-openrouter-key")
        
        # Should still return the key if it's the only one
        assert key_manager.get_next_available_key("openrouter") == "test-openrouter-key"
        
        # Add another key
        key_manager.add_api_key("openrouter", "test-openrouter-key-2")
        
        # Should return the new key since the other is rate limited
        assert key_manager.get_next_available_key("openrouter") == "test-openrouter-key-2"


if __name__ == "__main__":
    asyncio.run(pytest.main(["-xvs", __file__]))
