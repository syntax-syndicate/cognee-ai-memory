from typing import Optional, Type
import logging
from pydantic import BaseModel
import instructor
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.rate_limit import RateLimiter, RateLimitStrategy, StorageBackend
from cognee.infrastructure.llm.config import get_llm_config
import litellm
from cognee.infrastructure.llm.prompts import read_query_prompt

logger = logging.getLogger(__name__)

class AnthropicAdapter(LLMInterface):
    """Adapter for Anthropic's Claude API"""

    name = "Anthropic"
    model: str

    def __init__(self, max_tokens: int, model: str = "claude-3-opus-20240229"):
        self.model = model
        self.max_tokens = max_tokens
        self.aclient = instructor.from_litellm(
            litellm.acompletion, mode=instructor.Mode.JSON
        )
        
        # Initialize rate limiter
        self.config = get_llm_config()
        self.rate_limiter = None
        
        if self.config.llm_rate_limit_enabled:
            self._setup_rate_limiter(f"Anthropic-{model}")
    
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
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": text_input,
                },
            ],
            max_retries=5,
            response_model=response_model,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""

        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )

        return formatted_prompt
