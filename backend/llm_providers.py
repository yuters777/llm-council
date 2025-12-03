"""
Provider-agnostic LLM interface for direct API calls to OpenAI, Anthropic, and Google.

This module provides a unified interface for calling different LLM providers,
abstracting away the differences in their API formats.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)

# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# API Endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
RETRYABLE_STATUS_CODES = {429, 502, 503}

# Shared HTTP client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create the shared HTTP client.

    Returns:
        The shared AsyncClient instance with connection pooling.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        logger.debug("Created new HTTP client with connection pooling")
    return _http_client


async def close_http_client() -> None:
    """
    Close the shared HTTP client gracefully.

    Should be called during application shutdown.
    """
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        logger.debug("Closed HTTP client")
    _http_client = None


def _truncate_for_logging(text: str, max_length: int = 200) -> str:
    """Truncate text for safe logging."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    **kwargs
) -> httpx.Response:
    """
    Make an HTTP request with retry logic and exponential backoff.

    Args:
        client: The httpx AsyncClient to use
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        **kwargs: Additional arguments passed to client.request()

    Returns:
        The successful HTTP response

    Raises:
        httpx.HTTPStatusError: If request fails after all retries
        httpx.RequestError: If network error occurs after all retries
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)

            # Check if we got a retryable status code
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Received status %d from %s, retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    _truncate_for_logging(url),
                    delay,
                    attempt + 1,
                    max_retries
                )
                await asyncio.sleep(delay)
                continue

            # Raise for non-retryable error status codes
            response.raise_for_status()
            return response

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Timeout calling %s, retrying in %.1fs (attempt %d/%d)",
                    _truncate_for_logging(url),
                    delay,
                    attempt + 1,
                    max_retries
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Timeout calling %s after %d attempts", _truncate_for_logging(url), max_retries + 1)
                raise

        except httpx.RequestError as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Network error calling %s: %s, retrying in %.1fs (attempt %d/%d)",
                    _truncate_for_logging(url),
                    str(e),
                    delay,
                    attempt + 1,
                    max_retries
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Network error calling %s after %d attempts: %s",
                    _truncate_for_logging(url),
                    max_retries + 1,
                    str(e)
                )
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry loop")


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    provider: Literal["openai", "anthropic", "google"]
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096

    def __hash__(self):
        return hash((self.provider, self.model))

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            return False
        return self.provider == other.provider and self.model == other.model

    @property
    def display_name(self) -> str:
        """Return a human-readable name for this model."""
        return f"{self.provider}/{self.model}"


def validate_api_keys(configs: List[ModelConfig]) -> None:
    """
    Validate that required API keys are present for the configured models.

    Args:
        configs: List of ModelConfig to validate

    Raises:
        ValueError: If a required API key is missing
    """
    providers_needed = set(c.provider for c in configs)

    missing = []
    if "openai" in providers_needed and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if "anthropic" in providers_needed and not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if "google" in providers_needed and not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing)}. "
            "Please set them in your .env file."
        )


async def _call_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float
) -> Optional[str]:
    """
    Call OpenAI's Chat Completions API.

    Args:
        model: OpenAI model name (e.g., "gpt-4.1", "gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Response text or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug("Calling OpenAI model %s", model)

    try:
        client = await get_http_client()
        response = await _request_with_retry(
            client,
            "POST",
            OPENAI_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        data = response.json()
        content = data['choices'][0]['message']['content']
        logger.debug("OpenAI model %s returned %d characters", model, len(content) if content else 0)
        return content

    except httpx.HTTPStatusError as e:
        response_text = _truncate_for_logging(e.response.text)
        logger.error(
            "HTTP error from OpenAI model %s: status %d, response: %s",
            model,
            e.response.status_code,
            response_text
        )
        return None

    except httpx.TimeoutException:
        logger.error("Timeout calling OpenAI model %s after %.1fs", model, timeout)
        return None

    except httpx.RequestError as e:
        logger.error("Network error calling OpenAI model %s: %s", model, str(e))
        return None


async def _call_anthropic(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float
) -> Optional[str]:
    """
    Call Anthropic's Messages API.

    Args:
        model: Anthropic model name (e.g., "claude-3-5-sonnet-20241022")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Response text or None if failed
    """
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # Anthropic expects 'system' to be a separate field, not in messages
    # Convert our unified format to Anthropic's format
    system_content = None
    anthropic_messages = []

    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
        else:
            anthropic_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

    payload: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if system_content:
        payload["system"] = system_content

    logger.debug("Calling Anthropic model %s", model)

    try:
        client = await get_http_client()
        response = await _request_with_retry(
            client,
            "POST",
            ANTHROPIC_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        data = response.json()
        # Anthropic returns content as a list of content blocks
        content_blocks = data.get('content', [])
        if content_blocks and len(content_blocks) > 0:
            content = content_blocks[0].get('text', '')
            logger.debug("Anthropic model %s returned %d characters", model, len(content))
            return content
        logger.warning("Anthropic model %s returned empty content", model)
        return ''

    except httpx.HTTPStatusError as e:
        response_text = _truncate_for_logging(e.response.text)
        logger.error(
            "HTTP error from Anthropic model %s: status %d, response: %s",
            model,
            e.response.status_code,
            response_text
        )
        return None

    except httpx.TimeoutException:
        logger.error("Timeout calling Anthropic model %s after %.1fs", model, timeout)
        return None

    except httpx.RequestError as e:
        logger.error("Network error calling Anthropic model %s: %s", model, str(e))
        return None


async def _call_google(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float
) -> Optional[str]:
    """
    Call Google's Generative Language API (Gemini).

    Args:
        model: Gemini model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds

    Returns:
        Response text or None if failed
    """
    url = GOOGLE_API_URL_TEMPLATE.format(model=model)

    # Google uses 'contents' with 'parts' structure
    # Map roles: 'user' -> 'user', 'assistant' -> 'model', 'system' -> handled specially
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg['role']
        content = msg['content']

        if role == 'system':
            # Gemini handles system instructions differently
            system_instruction = content
        elif role == 'assistant':
            contents.append({
                'role': 'model',
                'parts': [{'text': content}]
            })
        else:  # user
            contents.append({
                'role': 'user',
                'parts': [{'text': content}]
            })

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    logger.debug("Calling Google model %s", model)

    try:
        client = await get_http_client()
        response = await _request_with_retry(
            client,
            "POST",
            url,
            params={"key": GOOGLE_API_KEY},
            json=payload,
            timeout=timeout
        )

        data = response.json()
        # Extract text from Gemini's response structure
        candidates = data.get('candidates', [])
        if candidates and len(candidates) > 0:
            content_data = candidates[0].get('content', {})
            parts = content_data.get('parts', [])
            if parts and len(parts) > 0:
                content = parts[0].get('text', '')
                logger.debug("Google model %s returned %d characters", model, len(content))
                return content
        logger.warning("Google model %s returned empty content", model)
        return ''

    except httpx.HTTPStatusError as e:
        response_text = _truncate_for_logging(e.response.text)
        logger.error(
            "HTTP error from Google model %s: status %d, response: %s",
            model,
            e.response.status_code,
            response_text
        )
        return None

    except httpx.TimeoutException:
        logger.error("Timeout calling Google model %s after %.1fs", model, timeout)
        return None

    except httpx.RequestError as e:
        logger.error("Network error calling Google model %s: %s", model, str(e))
        return None


async def call_model(
    model_config: ModelConfig,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[str]:
    """
    Call an LLM model using the appropriate provider.

    This is the main entry point for all LLM calls. It routes to the
    correct provider-specific implementation based on the model config.

    Args:
        model_config: Configuration specifying the provider and model
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response text as string, or None if the call failed
    """
    provider = model_config.provider
    model = model_config.model
    temperature = model_config.temperature
    max_tokens = model_config.max_tokens

    if provider == "openai":
        return await _call_openai(model, messages, temperature, max_tokens, timeout)
    elif provider == "anthropic":
        return await _call_anthropic(model, messages, temperature, max_tokens, timeout)
    elif provider == "google":
        return await _call_google(model, messages, temperature, max_tokens, timeout)
    else:
        logger.error("Unknown provider: %s", provider)
        return None


async def call_models_parallel(
    model_configs: List[ModelConfig],
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Dict[ModelConfig, Optional[str]]:
    """
    Call multiple models in parallel.

    Args:
        model_configs: List of ModelConfig for each model to call
        messages: List of message dicts to send to each model
        timeout: Request timeout in seconds

    Returns:
        Dict mapping ModelConfig to response string (or None if failed)
    """
    # Create tasks for all models
    tasks = [call_model(config, messages, timeout) for config in model_configs]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map configs to their responses
    return {config: response for config, response in zip(model_configs, responses)}
