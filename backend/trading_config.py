"""Configuration for the Trading Council feature.

Defines which models participate in trading analysis, their weights,
and operational parameters.
"""

import os

from dotenv import load_dotenv

from .llm_providers import ModelConfig

load_dotenv()

# Feature toggle — set TRADING_COUNCIL_ENABLED=false in .env to disable
TRADING_ENABLED: bool = os.getenv("TRADING_COUNCIL_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Council members with weighted votes
# Weights must sum to 1.0
TRADING_MODELS: dict[str, dict] = {
    "claude": {
        "weight": 0.40,
        "config": ModelConfig(
            provider="anthropic",
            model="claude-sonnet-4-6-20250415",
            max_tokens=1000,
            temperature=0.3,
        ),
    },
    "gpt": {
        "weight": 0.35,
        "config": ModelConfig(
            provider="openai",
            model="gpt-4.1",
            max_tokens=1000,
            temperature=0.3,
        ),
    },
    "gemini": {
        "weight": 0.25,
        "config": ModelConfig(
            provider="google",
            model="gemini-2.5-pro",
            max_tokens=1000,
            temperature=0.3,
        ),
    },
}

# Trigger types that skip Stage 2 peer review (time-sensitive)
SKIP_STAGE2_TRIGGERS: set[str] = {
    "OVERRIDE_STATE_CHANGE",
    "GEOSTRESS_ALERT",
    "UNUSUAL_PATTERN",
}

# Operational parameters
TRADING_TIMEOUT_SECONDS: float = 30.0
FALLBACK_DECISION: str = "HOLD"
FALLBACK_CONFIDENCE: float = 0.3
PROMPTS_DIR: str = "backend/prompts/trading"
