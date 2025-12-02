"""
Provider-agnostic LLM interface for direct API calls to OpenAI, Anthropic, and Google.

This module provides a unified interface for calling different LLM providers,
abstracting away the differences in their API formats.
"""

import os
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# API Endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


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

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            return data['choices'][0]['message']['content']

    except Exception as e:
        print(f"Error calling OpenAI model {model}: {e}")
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

    payload = {
        "model": model,
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if system_content:
        payload["system"] = system_content

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            # Anthropic returns content as a list of content blocks
            content_blocks = data.get('content', [])
            if content_blocks and len(content_blocks) > 0:
                return content_blocks[0].get('text', '')
            return ''

    except Exception as e:
        print(f"Error calling Anthropic model {model}: {e}")
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

    payload = {
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

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                params={"key": GOOGLE_API_KEY},
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            # Extract text from Gemini's response structure
            candidates = data.get('candidates', [])
            if candidates and len(candidates) > 0:
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if parts and len(parts) > 0:
                    return parts[0].get('text', '')
            return ''

    except Exception as e:
        print(f"Error calling Google model {model}: {e}")
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
        print(f"Unknown provider: {provider}")
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
    import asyncio

    # Create tasks for all models
    tasks = [call_model(config, messages, timeout) for config in model_configs]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map configs to their responses
    return {config: response for config, response in zip(model_configs, responses)}
