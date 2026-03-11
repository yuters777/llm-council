"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv
from .llm_providers import ModelConfig, validate_api_keys

load_dotenv()

# Council members - models that generate and evaluate responses
COUNCIL_MODELS = [
    ModelConfig(provider="openai", model="gpt-4.1"),
    ModelConfig(provider="anthropic", model="claude-sonnet-4-6-20250415"),
    ModelConfig(provider="google", model="gemini-2.0-flash"),
]

# Chairman model - synthesizes the final response
CHAIRMAN_MODEL = ModelConfig(provider="google", model="gemini-2.0-flash")

# Model used for generating conversation titles (fast and cheap)
TITLE_MODEL = ModelConfig(provider="google", model="gemini-2.0-flash", max_tokens=100)

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Validate API keys on import
def _validate_on_startup():
    """Validate that required API keys are present."""
    all_models = COUNCIL_MODELS + [CHAIRMAN_MODEL, TITLE_MODEL]
    validate_api_keys(all_models)

# Run validation (can be disabled for testing)
try:
    _validate_on_startup()
except ValueError as e:
    print(f"Warning: {e}")
