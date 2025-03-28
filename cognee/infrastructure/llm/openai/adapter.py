import os
import base64
import logging
from pathlib import Path
from typing import Type

import litellm
import instructor
from pydantic import BaseModel
from cognee.shared.data_models import MonitoringTool
from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.base_config import get_base_config
from cognee.infrastructure.llm.rate_limit import RateLimiter, RateLimitStrategy, StorageBackend
from cognee.infrastructure.llm.config import get_llm_config

logger = logging.getLogger(__name__)
monitoring = get_base_config().monitoring_tool

if monitoring == MonitoringTool.LANGFUSE:
    from langfuse.decorators import observe


class OpenAIAdapter(LLMInterface):
    name = "OpenAI"
    model: str
    api_key: str
    api_version: str

    MAX_RETRIES = 5

    """Adapter for OpenAI's GPT-3, GPT=4 API"""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: str,
        transcription_model: str,
        max_tokens: int,
        streaming: bool = False,
    ):
        self.aclient = instructor.from_litellm(litellm.acompletion)
        self.client = instructor.from_litellm(litellm.completion)
        self.transcription_model = transcription_model
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.streaming = streaming
        
        # Initialize rate limiter
        self.config = get_llm_config()
        self.rate_limiter = None
        
        if self.config.llm_rate_limit_enabled:
            self._setup_rate_limiter(f"OpenAI-{model}")
    
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

    @observe(as_type="generation")
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
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            response_model=response_model,
            max_retries=self.MAX_RETRIES,
        )

    @observe
    def create_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""
        # Check rate limit before making the API call
        if not self.check_rate_limit():
            raise InvalidValueError(message="Rate limit exceeded. Please try again later.")

        return self.client.chat.completions.create(
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
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            response_model=response_model,
            max_retries=self.MAX_RETRIES,
        )

    def create_transcript(self, input):
        """Generate a audio transcript from a user query."""
        # Check rate limit before making the API call
        if not self.check_rate_limit():
            raise InvalidValueError(message="Rate limit exceeded. Please try again later.")

        if not os.path.isfile(input):
            raise FileNotFoundError(f"The file {input} does not exist.")

        # with open(input, 'rb') as audio_file:
        #     audio_data = audio_file.read()

        transcription = litellm.transcription(
            model=self.transcription_model,
            file=Path(input),
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_retries=self.MAX_RETRIES,
        )

        return transcription

    def transcribe_image(self, input) -> BaseModel:
        # Check rate limit before making the API call
        if not self.check_rate_limit():
            raise InvalidValueError(message="Rate limit exceeded. Please try again later.")
            
        with open(input, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return litellm.completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            max_tokens=300,
            max_retries=self.MAX_RETRIES,
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
