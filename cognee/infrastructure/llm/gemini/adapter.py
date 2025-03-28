from typing import Type, Optional
import logging
from pydantic import BaseModel
from cognee.shared.logging_utils import get_logger
import litellm
from litellm import acompletion, JSONSchemaValidationError
from cognee.shared.data_models import MonitoringTool
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.base_config import get_base_config
from cognee.infrastructure.llm.rate_limit import RateLimiter, RateLimitStrategy, StorageBackend
from cognee.infrastructure.llm.config import get_llm_config

logger = get_logger()

monitoring = get_base_config().monitoring_tool

if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


class GeminiAdapter(LLMInterface):
    MAX_RETRIES = 5

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.api_version = api_version
        self.streaming = streaming
        self.max_tokens = max_tokens
        
        # Initialize rate limiter
        self.config = get_llm_config()
        self.rate_limiter = None
        
        if self.config.llm_rate_limit_enabled:
            self._setup_rate_limiter(f"Gemini-{model}")
    
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

    @observe(as_type="generation") if monitoring == MonitoringTool.LANGFUSE else lambda f: f
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Generate a structured output from a user query using Gemini model.
        This approach uses schema-based guidance for the model.
        """
        # Check rate limit before making the API call
        if not await self.acheck_rate_limit():
            raise InvalidValueError(message="Rate limit exceeded. Please try again later.")

        prompt = f"System instruction: {system_prompt}\n\nUser query: {text_input}"

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                api_base=self.endpoint,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                max_retries=self.MAX_RETRIES,
            )

            message_content = response.choices[0].message.content
            # Attempt to parse the generated content into the response model
            try:
                result = response_model.model_validate_json(message_content)
                return result
            except Exception as e:
                logger.error(f"Failed to parse Gemini response into model: {str(e)}")
                # Fallback: try to extract as much as possible into a partial object
                import json
                try:
                    data = json.loads(message_content)
                    # Create partial model with whatever fields we can extract
                    result = response_model.model_validate(data)
                    return result
                except Exception as json_error:
                    logger.error(f"Failed JSON parsing: {str(json_error)}")
                    raise ValueError(f"Failed to parse Gemini response: {message_content}") from e

        except JSONSchemaValidationError as e:
            logger.error(f"JSON schema validation error: {str(e)}")
            raise InvalidValueError(message=f"Invalid JSON response from Gemini: {str(e)}")
        except Exception as e:
            logger.error(f"Error calling Gemini: {str(e)}")
            raise InvalidValueError(message=f"Error from Gemini API: {str(e)}")

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""
        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)
        formatted_prompt = f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
        return formatted_prompt
