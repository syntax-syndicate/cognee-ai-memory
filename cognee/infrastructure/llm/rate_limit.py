from typing import Optional, Union, Any, Type, cast
import logging
from abc import ABC, abstractmethod
from enum import Enum
import limits.strategies
from limits.aio.strategies import MovingWindowRateLimiter as AsyncMovingWindowRateLimiter
from limits.aio.strategies import FixedWindowRateLimiter as AsyncFixedWindowRateLimiter
from limits.aio.storage import Storage as AsyncStorage
from limits.storage import Storage, MemoryStorage, RedisStorage, MemcachedStorage
from limits.limits import RateLimitItemPerSecond, RateLimitItemPerMinute, RateLimitItemPerHour, RateLimitItemPerDay
from limits.errors import ConfigurationError

logger = logging.getLogger(__name__)

class RateLimitStrategy(str, Enum):
    """Rate limiting strategies supported by the system."""
    FIXED_WINDOW = "fixed-window"
    MOVING_WINDOW = "moving-window"
    SLIDING_WINDOW = "sliding-window"

class StorageBackend(str, Enum):
    """Storage backends for rate limiting data."""
    MEMORY = "memory"
    REDIS = "redis" 
    MEMCACHED = "memcached"

class RateLimiter:
    """Rate limiter implementation using the 'limits' library."""
    
    def __init__(
        self,
        rate_limit: str,
        resource_name: str,
        strategy: RateLimitStrategy = RateLimitStrategy.MOVING_WINDOW,
        storage_backend: StorageBackend = StorageBackend.MEMORY,
        redis_url: Optional[str] = None,
        memcached_host: Optional[str] = None,
        memcached_port: int = 11211,
    ):
        """
        Initialize a rate limiter.
        
        Args:
            rate_limit: Rate limit in string notation (e.g. "100/minute", "10/second")
            resource_name: Name of the resource to rate limit
            strategy: Rate limiting strategy to use
            storage_backend: Storage backend for rate limiting data
            redis_url: Redis URL (required if storage_backend is REDIS)
            memcached_host: Memcached host (required if storage_backend is MEMCACHED)
            memcached_port: Memcached port (default: 11211)
        """
        self.rate_limit = rate_limit
        self.resource_name = resource_name
        
        # Initialize storage
        self.storage = self._get_storage(storage_backend, redis_url, memcached_host, memcached_port)
        
        # Initialize strategy
        self.strategy = self._get_strategy(strategy)
        
        # Initialize async strategy
        self.async_strategy = self._get_async_strategy(strategy)
        
        logger.info(f"Rate limiter initialized for {resource_name} with {rate_limit} using {strategy} strategy")
    
    def _get_storage(
        self, 
        storage_backend: StorageBackend,
        redis_url: Optional[str] = None,
        memcached_host: Optional[str] = None,
        memcached_port: int = 11211
    ) -> Storage:
        """Get the appropriate storage backend."""
        try:
            if storage_backend == StorageBackend.MEMORY:
                return MemoryStorage()
            elif storage_backend == StorageBackend.REDIS:
                if not redis_url:
                    raise ConfigurationError("Redis URL is required for Redis storage backend")
                return RedisStorage(redis_url)
            elif storage_backend == StorageBackend.MEMCACHED:
                if not memcached_host:
                    raise ConfigurationError("Memcached host is required for Memcached storage backend")
                return MemcachedStorage(f"{memcached_host}:{memcached_port}")
            else:
                raise ValueError(f"Unsupported storage backend: {storage_backend}")
        except ImportError as e:
            # If dependencies aren't available, fall back to memory storage for tests
            logger.warning(f"Storage dependency not available: {e}. Falling back to memory storage.")
            return MemoryStorage()
    
    def _get_strategy(self, strategy: RateLimitStrategy):
        """Get the appropriate rate limiting strategy."""
        if strategy == RateLimitStrategy.FIXED_WINDOW:
            return limits.strategies.FixedWindowRateLimiter(self.storage)
        elif strategy == RateLimitStrategy.MOVING_WINDOW:
            return limits.strategies.MovingWindowRateLimiter(self.storage)
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            return limits.strategies.FixedWindowElasticExpiryRateLimiter(self.storage)
        else:
            raise ValueError(f"Unsupported rate limiting strategy: {strategy}")
    
    def _get_async_strategy(self, strategy: RateLimitStrategy):
        """Get the appropriate async rate limiting strategy."""
        # Ensure the storage is properly cast to AsyncStorage for the async strategies
        storage = cast(AsyncStorage, self.storage)
        
        if strategy == RateLimitStrategy.FIXED_WINDOW:
            return AsyncFixedWindowRateLimiter(storage)
        elif strategy == RateLimitStrategy.MOVING_WINDOW:
            return AsyncMovingWindowRateLimiter(storage)
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            # Note: Currently there's no async version of sliding window, so we use fixed window as fallback
            logger.warning("Async sliding window not available, using fixed window instead")
            return AsyncFixedWindowRateLimiter(storage)
        else:
            raise ValueError(f"Unsupported rate limiting strategy: {strategy}")

    def hit(self) -> bool:
        """
        Record a hit and check if the request is allowed.
        
        Returns:
            bool: True if the request is allowed, False otherwise
        """
        return self.strategy.hit(self.rate_limit, self.resource_name)
    
    async def async_hit(self) -> bool:
        """
        Record a hit asynchronously and check if the request is allowed.
        
        Returns:
            bool: True if the request is allowed, False otherwise
        """
        return await self.async_strategy.hit(self.rate_limit, self.resource_name)
    
    def test(self) -> bool:
        """
        Check if the request would be allowed without recording a hit.
        
        Returns:
            bool: True if the request would be allowed, False otherwise
        """
        return self.strategy.test(self.rate_limit, self.resource_name)
    
    async def async_test(self) -> bool:
        """
        Check asynchronously if the request would be allowed without recording a hit.
        
        Returns:
            bool: True if the request would be allowed, False otherwise
        """
        return await self.async_strategy.test(self.rate_limit, self.resource_name)
    
    def get_window_stats(self) -> tuple:
        """
        Get the current window statistics.
        
        Returns:
            tuple: (current_window_count, current_limit)
        """
        # This is only available for some strategies
        if hasattr(self.strategy, "get_window_stats"):
            return self.strategy.get_window_stats(self.rate_limit, self.resource_name)
        else:
            # For strategies that don't support this, return a reasonable default
            return (0, int(self.rate_limit.split("/")[0]))
    
    def reset(self) -> None:
        """Reset the rate limiter for the resource."""
        self.storage.clear(self.resource_name) 