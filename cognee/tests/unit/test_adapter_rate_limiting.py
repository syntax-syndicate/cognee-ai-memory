"""Unit tests for LLM adapter rate limiting functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.openai.adapter import OpenAIAdapter
from cognee.infrastructure.llm.gemini.adapter import GeminiAdapter
from cognee.infrastructure.llm.rate_limit import RateLimiter


@pytest.fixture
def mock_config():
    """Mock LLM config with rate limiting enabled."""
    config = MagicMock()
    config.llm_rate_limit_enabled = True
    config.llm_rate_limit = "2/minute"
    config.llm_rate_limit_strategy = "fixed-window"
    config.llm_rate_limit_storage = "memory"
    config.llm_rate_limit_redis_url = None
    config.llm_rate_limit_memcached_host = None
    config.llm_rate_limit_memcached_port = 11211
    return config


@pytest.fixture
def mock_litellm():
    """Mock LiteLLM for testing."""
    mock = MagicMock()
    mock.completion = MagicMock()
    mock.acompletion = AsyncMock()
    return mock


@pytest.fixture
def mock_instructor():
    """Mock Instructor for testing."""
    mock = MagicMock()
    mock.from_litellm.return_value = MagicMock()
    return mock


class TestOpenAIAdapterRateLimiting:
    """Test rate limiting in the OpenAI adapter."""

    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    def test_init_rate_limiter(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test initialization of rate limiter in OpenAI adapter."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        assert adapter.rate_limiter is not None
        assert adapter.rate_limiter.resource_name == "OpenAI-gpt-4"
        assert adapter.rate_limiter.rate_limit == "2/minute"

    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    def test_check_rate_limit_success(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test successful rate limit check."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        # Mock the rate limiter
        adapter.rate_limiter = MagicMock(spec=RateLimiter)
        adapter.rate_limiter.hit.return_value = True
        
        # Check should succeed
        assert adapter.check_rate_limit() is True
        adapter.rate_limiter.hit.assert_called_once()

    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    def test_check_rate_limit_failure(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test rate limit exceeded."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        # Mock the rate limiter
        adapter.rate_limiter = MagicMock(spec=RateLimiter)
        adapter.rate_limiter.hit.return_value = False
        adapter.rate_limiter.get_window_stats.return_value = (3, 2)
        
        # Check should fail
        assert adapter.check_rate_limit() is False
        adapter.rate_limiter.hit.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    async def test_async_check_rate_limit_success(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test successful async rate limit check."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        # Mock the rate limiter
        adapter.rate_limiter = MagicMock(spec=RateLimiter)
        adapter.rate_limiter.async_hit = AsyncMock(return_value=True)
        
        # Async check should succeed
        assert await adapter.acheck_rate_limit() is True
        adapter.rate_limiter.async_hit.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    async def test_async_check_rate_limit_failure(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test async rate limit exceeded."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        # Mock the rate limiter
        adapter.rate_limiter = MagicMock(spec=RateLimiter)
        adapter.rate_limiter.async_hit = AsyncMock(return_value=False)
        adapter.rate_limiter.get_window_stats.return_value = (3, 2)
        
        # Async check should fail
        assert await adapter.acheck_rate_limit() is False
        adapter.rate_limiter.async_hit.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.openai.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.openai.adapter.instructor")
    @patch("cognee.infrastructure.llm.openai.adapter.litellm")
    async def test_acreate_structured_output_rate_limited(self, mock_litellm, mock_instructor, mock_get_config, mock_config):
        """Test acreate_structured_output with rate limiting."""
        mock_get_config.return_value = mock_config
        
        adapter = OpenAIAdapter(
            api_key="test-key",
            endpoint="https://api.openai.com",
            api_version="v1",
            model="gpt-4",
            transcription_model="whisper-1",
            max_tokens=1000,
        )
        
        # Mock the rate limiter to reject the request
        adapter.acheck_rate_limit = AsyncMock(return_value=False)
        
        # Call should raise an exception
        with pytest.raises(InvalidValueError, match="Rate limit exceeded"):
            await adapter.acreate_structured_output("test input", "test prompt", dict)

        # Confirm rate limit was checked
        adapter.acheck_rate_limit.assert_called_once()
        
        # The API call should not have been made
        adapter.aclient.chat.completions.create.assert_not_called()


class TestGeminiAdapterRateLimiting:
    """Test rate limiting in the Gemini adapter."""
    
    @patch("cognee.infrastructure.llm.gemini.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.gemini.adapter.litellm")
    def test_init_rate_limiter(self, mock_litellm, mock_get_config, mock_config):
        """Test initialization of rate limiter in Gemini adapter."""
        mock_get_config.return_value = mock_config
        
        adapter = GeminiAdapter(
            api_key="test-key",
            model="gemini-pro",
            max_tokens=1000,
        )
        
        assert adapter.rate_limiter is not None
        assert adapter.rate_limiter.resource_name == "Gemini-gemini-pro"
        assert adapter.rate_limiter.rate_limit == "2/minute"
    
    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.gemini.adapter.get_llm_config")
    @patch("cognee.infrastructure.llm.gemini.adapter.litellm")
    @patch("cognee.infrastructure.llm.gemini.adapter.acompletion")
    async def test_acreate_structured_output_rate_limited(self, mock_acompletion, mock_litellm, mock_get_config, mock_config):
        """Test acreate_structured_output with rate limiting for Gemini."""
        mock_get_config.return_value = mock_config
        
        adapter = GeminiAdapter(
            api_key="test-key",
            model="gemini-pro",
            max_tokens=1000,
        )
        
        # Mock the rate limiter to reject the request
        adapter.acheck_rate_limit = AsyncMock(return_value=False)
        
        # Call should raise an exception
        with pytest.raises(InvalidValueError, match="Rate limit exceeded"):
            await adapter.acreate_structured_output("test input", "test prompt", dict)

        # Confirm rate limit was checked
        adapter.acheck_rate_limit.assert_called_once()
        
        # The API call should not have been made
        mock_acompletion.assert_not_called() 