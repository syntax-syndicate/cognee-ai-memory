"""Adapter for Generic API LLM provider API"""

from typing import Type
import logging
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.config import get_llm_config
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.rate_limit import RateLimiter, RateLimitStrategy, StorageBackend
import litellm

logger = logging.getLogger(__name__)

class GenericAPIAdapter(LLMInterface):
    """Adapter for Generic API LLM provider API"""

    name: str
    model: str
    api_key: str

    def __init__(self, endpoint, api_key: str, model: str, name: str, max_tokens: int):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens

        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode.JSON, api_key=api_key
        )
        
        # Initialize rate limiter
        self.config = get_llm_config()
        self.rate_limiter = None
        
        if self.config.llm_rate_limit_enabled:
            self._setup_rate_limiter(f"{name}-{model}")
    
    def _setup_rate_limiter(self, resource_name: str):
        """Set up the rate limiter based on configuration."""
        try:
            # Convert string configs to enum values
            strategy = RateLimitStrategy(self.config.llm_rate_limit_strategy)
            storage = StorageBackend(self.config.llm_rate_limit_storage)
            
            # Create rate limiter
            self.rate_limiter = RateLimiter(
                rate_limit=self.config.llm_rate_limit,
                resource_name=resource_name,
                strategy=strategy,
                storage_backend=storage,
                redis_url=self.config.llm_rate_limit_redis_url,
                memcached_host=self.config.llm_rate_limit_memcached_host,
                memcached_port=self.config.llm_rate_limit_memcached_port,
            )
            
            logger.info(
                f"Rate limiter initialized for {resource_name} with limit {self.config.llm_rate_limit}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {str(e)}")
            self.rate_limiter = None

    def check_rate_limit(self) -> bool:
        """Check if the current request is allowed by the rate limiter."""
        if not self.config.llm_rate_limit_enabled or self.rate_limiter is None:
            return True
            
        is_allowed = self.rate_limiter.hit()
        
        if not is_allowed:
            window_stats = self.rate_limiter.get_window_stats()
            logger.warning(
                f"Rate limit exceeded: {window_stats[0]}/{window_stats[1]} "
                f"requests for {self.rate_limiter.resource_name}"
            )
            
        return is_allowed
    
    async def acheck_rate_limit(self) -> bool:
        """Asynchronously check if the current request is allowed by the rate limiter."""
        if not self.config.llm_rate_limit_enabled or self.rate_limiter is None:
            return True
            
        is_allowed = await self.rate_limiter.async_hit()
        
        if not is_allowed:
            window_stats = self.rate_limiter.get_window_stats()
            logger.warning(
                f"Rate limit exceeded: {window_stats[0]}/{window_stats[1]} "
                f"requests for {self.rate_limiter.resource_name}"
            )
            
        return is_allowed

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""
        # Check rate limit before making the API call
        if not await self.acheck_rate_limit():
            raise InvalidValueError(message="Rate limit exceeded. Please try again later.")

        return await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to
                extract information from the following input: {text_input}. """,
                },
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            max_retries=5,
            api_base=self.endpoint,
            response_model=response_model,
        )
