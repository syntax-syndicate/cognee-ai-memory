"""Unit tests for the rate limiter functionality."""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, Mock
from limits.aio.storage import Storage as AsyncStorage
from cognee.infrastructure.llm.rate_limit import (
    RateLimiter,
    RateLimitStrategy,
    StorageBackend,
)


class TestRateLimiter:
    """Test suite for the RateLimiter class."""

    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_init_memory_storage(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test initialization with memory storage."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        mock_strategy.return_value = MagicMock()
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="10/second",
            resource_name="test-resource",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        assert limiter.rate_limit == "10/second"
        assert limiter.resource_name == "test-resource"
        mock_memory_storage.assert_called_once()
        
    @patch("cognee.infrastructure.llm.rate_limit.RedisStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_init_redis_storage(self, mock_async_strategy, mock_strategy, mock_redis_storage):
        """Test initialization with Redis storage."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_redis_storage.return_value = mock_storage_instance
        mock_strategy.return_value = MagicMock()
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="10/second",
            resource_name="test-resource",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.REDIS,
            redis_url="redis://localhost:6379/0",
        )
        
        assert limiter.rate_limit == "10/second"
        assert limiter.resource_name == "test-resource"
        mock_redis_storage.assert_called_once_with("redis://localhost:6379/0")
        
    @patch("cognee.infrastructure.llm.rate_limit.MemcachedStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_init_memcached_storage(self, mock_async_strategy, mock_strategy, mock_memcached_storage):
        """Test initialization with Memcached storage."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memcached_storage.return_value = mock_storage_instance
        mock_strategy.return_value = MagicMock()
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="10/second",
            resource_name="test-resource",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMCACHED,
            memcached_host="localhost",
            memcached_port=11211,
        )
        
        assert limiter.rate_limit == "10/second"
        assert limiter.resource_name == "test-resource"
        mock_memcached_storage.assert_called_once_with("localhost:11211")
        
    def test_init_invalid_storage(self):
        """Test initialization with an invalid storage backend."""
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            RateLimiter(
                rate_limit="10/second",
                resource_name="test-resource",
                strategy=RateLimitStrategy.FIXED_WINDOW,
                storage_backend="invalid",
            )
            
    def test_init_invalid_strategy(self):
        """Test initialization with an invalid strategy."""
        with pytest.raises(ValueError, match="Unsupported rate limiting strategy"):
            RateLimiter(
                rate_limit="10/second",
                resource_name="test-resource",
                strategy="invalid",
                storage_backend=StorageBackend.MEMORY,
            )
    
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_fixed_window_strategy(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test the fixed window rate limiting strategy."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the hit method to return True twice, then False, then True
        mock_strategy_instance.hit.side_effect = [True, True, False, True]
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-fixed-window",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # First two hits should succeed
        assert limiter.hit() is True
        assert limiter.hit() is True
        
        # Third hit should fail
        assert limiter.hit() is False
        
        # Wait for a new window
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.hit() is True
        
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.MovingWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncMovingWindowRateLimiter")
    def test_moving_window_strategy(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test the moving window rate limiting strategy."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the hit method to return True twice, then False, then True
        mock_strategy_instance.hit.side_effect = [True, True, False, True]
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-moving-window",
            strategy=RateLimitStrategy.MOVING_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # First two hits should succeed
        assert limiter.hit() is True
        assert limiter.hit() is True
        
        # Third hit should fail
        assert limiter.hit() is False
        
        # Wait for a new window
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.hit() is True
        
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowElasticExpiryRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_sliding_window_strategy(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test the sliding window rate limiting strategy."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the hit method to return True twice, then False, then True
        mock_strategy_instance.hit.side_effect = [True, True, False, True]
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-sliding-window",
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # First two hits should succeed
        assert limiter.hit() is True
        assert limiter.hit() is True
        
        # Third hit should fail
        assert limiter.hit() is False
        
        # Wait for a new window
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.hit() is True
        
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.MovingWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncMovingWindowRateLimiter")
    def test_window_stats(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test getting window statistics."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the get_window_stats method to return appropriate values
        mock_strategy_instance.get_window_stats.side_effect = [(0, 10), (3, 10)]
        
        # Configure hit method
        mock_strategy_instance.hit.return_value = True
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="10/second",
            resource_name="test-window-stats",
            strategy=RateLimitStrategy.MOVING_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # Before any hits
        current, limit = limiter.get_window_stats()
        assert current == 0
        assert limit == 10
        
        # After some hits
        limiter.hit()
        limiter.hit()
        limiter.hit()
        
        current, limit = limiter.get_window_stats()
        assert current == 3
        assert limit == 10
        
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_reset(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test resetting the rate limiter."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the hit method to return True, True, False, True
        mock_strategy_instance.hit.side_effect = [True, True, False, True]
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-reset",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # Use up the limit
        limiter.hit()
        limiter.hit()
        assert limiter.hit() is False
        
        # Reset and should be allowed again
        limiter.reset()
        assert limiter.hit() is True
        
    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    async def test_async_hit(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test async_hit method."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        mock_async_strategy_instance = MagicMock()
        mock_async_strategy.return_value = mock_async_strategy_instance
        
        # Configure the async_hit method using simpler approach
        future1 = asyncio.Future()
        future1.set_result(True)
        future2 = asyncio.Future()
        future2.set_result(True)
        future3 = asyncio.Future()
        future3.set_result(False)
        future4 = asyncio.Future()
        future4.set_result(True)
        
        mock_async_strategy_instance.hit.side_effect = [future1, future2, future3, future4]
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-async-hit",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # First two hits should succeed
        assert await limiter.async_hit() is True
        assert await limiter.async_hit() is True
        
        # Third hit should fail
        assert await limiter.async_hit() is False
        
        # Wait for a new window
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        assert await limiter.async_hit() is True
        
    @pytest.mark.asyncio
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    async def test_async_test(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test async_test method."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        mock_async_strategy_instance = MagicMock()
        mock_async_strategy.return_value = mock_async_strategy_instance
        
        # Configure the async_test method using simpler approach
        test_future1 = asyncio.Future()
        test_future1.set_result(True)
        test_future2 = asyncio.Future()
        test_future2.set_result(False)
        test_future3 = asyncio.Future()
        test_future3.set_result(True)
        
        mock_async_strategy_instance.test.side_effect = [test_future1, test_future2, test_future3]
        
        # Configure the async_hit method
        hit_future1 = asyncio.Future()
        hit_future1.set_result(True)
        hit_future2 = asyncio.Future()
        hit_future2.set_result(True)
        
        mock_async_strategy_instance.hit.side_effect = [hit_future1, hit_future2]
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-async-test",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # Before any hits
        assert await limiter.async_test() is True
        
        # Use up the limit
        await limiter.async_hit()
        await limiter.async_hit()
        
        # Test should return False but not record a hit
        assert await limiter.async_test() is False
        
        # Wait for a new window
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        assert await limiter.async_test() is True
        
    @patch("cognee.infrastructure.llm.rate_limit.MemoryStorage")
    @patch("cognee.infrastructure.llm.rate_limit.limits.strategies.FixedWindowRateLimiter")
    @patch("cognee.infrastructure.llm.rate_limit.AsyncFixedWindowRateLimiter")
    def test_test_method(self, mock_async_strategy, mock_strategy, mock_memory_storage):
        """Test the test method."""
        # Set up mocks
        mock_storage_instance = MagicMock()
        mock_memory_storage.return_value = mock_storage_instance
        
        mock_strategy_instance = MagicMock()
        mock_strategy.return_value = mock_strategy_instance
        
        # Configure the test method
        mock_strategy_instance.test.side_effect = [True, False, True]
        
        # Configure hit method
        mock_strategy_instance.hit.return_value = True
        
        mock_async_strategy.return_value = MagicMock()
        
        limiter = RateLimiter(
            rate_limit="2/second",
            resource_name="test-test-method",
            strategy=RateLimitStrategy.FIXED_WINDOW,
            storage_backend=StorageBackend.MEMORY,
        )
        
        # Before any hits
        assert limiter.test() is True
        
        # Use up the limit
        limiter.hit()
        limiter.hit()
        
        # Test should return False but not record a hit
        assert limiter.test() is False
        
        # Wait for a new window
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.test() is True 