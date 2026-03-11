"""Core trading council logic.

Orchestrates prompt building, parallel model calls, response parsing,
consensus aggregation, and Telegram alert formatting.

All LLM calls go through llm_providers.call_models_parallel() — never
import httpx or call APIs directly from this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from .llm_providers import ModelConfig, call_models_parallel
from .trading_config import (
    FALLBACK_CONFIDENCE,
    FALLBACK_DECISION,
    PROMPTS_DIR,
    SKIP_STAGE2_TRIGGERS,
    TRADING_MODELS,
    TRADING_TIMEOUT_SECONDS,
)
from .trading_models import (
    ConsensusResult,
    MarketSnapshot,
    ModelVote,
    TradingDecision,
    TradingMeta,
)

logger = logging.getLogger(__name__)

# ── Module-level caches ──────────────────────────────────────────────────────

_template_cache: dict[str, str] = {}

# Trigger type → prompt template filename (without .md)
_TRIGGER_FILE_MAP: dict[str, str] = {
    "OVERRIDE_STATE_CHANGE": "override_change",
    "GEOSTRESS_ALERT": "geostress_alert",
    "CONFLICTING_SIGNALS": "conflicting_signals",
    "MORNING_BRIEFING": "morning_briefing",
    "UNUSUAL_PATTERN": "unusual_pattern",
    "EARNINGS_PROXIMITY": "unusual_pattern",  # reuse unusual_pattern template
}

# Emoji prefixes for Telegram alerts
_TRIGGER_EMOJI: dict[str, str] = {
    "OVERRIDE_STATE_CHANGE": "\U0001f7e2",   # green circle
    "GEOSTRESS_ALERT": "\U0001f6a8",          # rotating light
    "CONFLICTING_SIGNALS": "\u26a0\ufe0f",    # warning
    "MORNING_BRIEFING": "\U0001f4cb",          # clipboard
    "UNUSUAL_PATTERN": "\U0001f50d",           # magnifying glass
    "EARNINGS_PROXIMITY": "\U0001f4c5",        # calendar
}


# ── 1. Prompt Building ───────────────────────────────────────────────────────

def load_prompt_template(trigger_type: str) -> str:
    """Load the user-message template for a given trigger type.

    Templates are cached after first load.

    Raises:
        ValueError: If trigger_type is not recognised.
        FileNotFoundError: If the template file doesn't exist.
    """
    if trigger_type in _template_cache:
        return _template_cache[trigger_type]

    filename = _TRIGGER_FILE_MAP.get(trigger_type)
    if filename is None:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

    path = Path(PROMPTS_DIR) / f"{filename}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    content = path.read_text()
    _template_cache[trigger_type] = content
    return content


def load_system_prompt() -> str:
    """Load and cache the base system prompt."""
    cache_key = "__system__"
    if cache_key in _template_cache:
        return _template_cache[cache_key]

    path = Path(PROMPTS_DIR) / "system_base.md"
    content = path.read_text()
    _template_cache[cache_key] = content
    return content


def build_prompts(snapshot: MarketSnapshot) -> tuple[str, list[dict]]:
    """Build the system prompt and user messages for the council call.

    Returns:
        (system_prompt, messages) where messages includes the system message
        and a single user message with the snapshot data injected.
    """
    system_prompt = load_system_prompt()
    template = load_prompt_template(snapshot.trigger_type)

    snapshot_json = snapshot.model_dump_json(indent=2)

    if snapshot.trigger_type == "MORNING_BRIEFING":
        user_content = template.replace("{snapshot_json}", snapshot_json)
        user_content = user_content.replace("{news_json}", "{}")
        user_content = user_content.replace("{movers_json}", "{}")
    else:
        user_content = template.replace("{snapshot_json}", snapshot_json)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return system_prompt, messages


# ── 2. Response Parsing (3-tier fallback) ─────────────────────────────────────

def parse_trading_response(raw_text: str, model_name: str) -> ModelVote:
    """Parse a model's raw text response into a ModelVote.

    Tier 1: Direct JSON parse
    Tier 2: Extract JSON from ```json ... ``` markdown block
    Tier 3: Regex extraction of individual fields
    Fallback: HOLD with low confidence (safety net)
    """
    if raw_text is None:
        logger.warning("Model %s returned None response", model_name)
        return ModelVote(
            decision=FALLBACK_DECISION,
            confidence=FALLBACK_CONFIDENCE,
            key_factor="No response received",
            risk_flag="Model returned no output",
            reasoning="Model failed to respond",
        )

    # Tier 1: Direct JSON parse
    try:
        data = json.loads(raw_text.strip())
        logger.debug("Model %s: parsed via Tier 1 (direct JSON)", model_name)
        return _dict_to_model_vote(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: Extract from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1).strip())
            logger.debug("Model %s: parsed via Tier 2 (markdown block)", model_name)
            return _dict_to_model_vote(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: Regex extraction of individual fields
    decision_match = re.search(r'"decision"\s*:\s*"([A-Z_]+)"', raw_text)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
    key_factor_match = re.search(r'"key_factor"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
    risk_flag_match = re.search(r'"risk_flag"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)

    if decision_match and confidence_match:
        logger.debug("Model %s: parsed via Tier 3 (regex)", model_name)
        return ModelVote(
            decision=decision_match.group(1),
            confidence=_clamp_confidence(float(confidence_match.group(1))),
            key_factor=key_factor_match.group(1) if key_factor_match else "",
            risk_flag=risk_flag_match.group(1) if risk_flag_match else None,
            reasoning=reasoning_match.group(1) if reasoning_match else "",
        )

    # Fallback: HOLD with low confidence
    logger.warning(
        "Model %s: all parse tiers failed, using fallback HOLD",
        model_name,
    )
    return ModelVote(
        decision=FALLBACK_DECISION,
        confidence=FALLBACK_CONFIDENCE,
        key_factor="Parse failed",
        risk_flag="Raw response could not be parsed",
        reasoning=raw_text[:500],
    )


def _dict_to_model_vote(data: dict) -> ModelVote:
    """Convert a parsed dict to a ModelVote with safe defaults."""
    return ModelVote(
        decision=data.get("decision", FALLBACK_DECISION),
        confidence=_clamp_confidence(float(data.get("confidence", 0.5))),
        key_factor=data.get("key_factor", ""),
        risk_flag=data.get("risk_flag"),
        reasoning=data.get("reasoning", ""),
    )


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ── 3. Consensus Aggregation ─────────────────────────────────────────────────

def aggregate_decisions(votes: dict[str, ModelVote]) -> ConsensusResult:
    """Aggregate individual model votes into a consensus result.

    - Majority vote wins; on 3-way split the decision falls back to HOLD.
    - Confidence is weighted by model weights from TRADING_MODELS config.
    - Dissenters are identified and summarised.
    """
    if not votes:
        return ConsensusResult(
            decision="HOLD",
            confidence=FALLBACK_CONFIDENCE,
            consensus="No votes received",
            consensus_strength="0/3",
            dissent_summary="No models responded",
        )

    # Count decisions
    decision_counts: Counter[str] = Counter()
    for vote in votes.values():
        decision_counts[vote.decision] += 1

    total_models = len(votes)
    most_common_decision, most_common_count = decision_counts.most_common(1)[0]

    # 3-way split → HOLD
    if most_common_count == 1 and total_models >= 3:
        final_decision = "HOLD"
    else:
        final_decision = most_common_decision

    # Weighted confidence
    weighted_confidence = 0.0
    total_weight = 0.0
    for model_name, vote in votes.items():
        weight = TRADING_MODELS.get(model_name, {}).get("weight", 1.0 / total_models)
        weighted_confidence += vote.confidence * weight
        total_weight += weight

    if total_weight > 0:
        weighted_confidence /= total_weight
        weighted_confidence *= total_weight  # sum, not average per spec

    # Consensus strength
    if most_common_count == total_models:
        consensus_strength = "UNANIMOUS"
    else:
        consensus_strength = f"{most_common_count}/{total_models}"

    # Dissent
    dissenters: list[str] = []
    for model_name, vote in votes.items():
        if vote.decision != final_decision:
            dissenters.append(
                f"{model_name} suggests {vote.decision} based on {vote.key_factor}"
            )

    dissent_summary = "; ".join(dissenters) if dissenters else ""
    consensus_label = f"{consensus_strength}_{final_decision}"

    return ConsensusResult(
        decision=final_decision,
        confidence=round(weighted_confidence, 3),
        consensus=consensus_label,
        consensus_strength=consensus_strength,
        dissent_summary=dissent_summary,
    )


# ── 4. Alert Formatting ──────────────────────────────────────────────────────

def format_telegram_alert(
    snapshot: MarketSnapshot,
    result: ConsensusResult,
    votes: dict[str, ModelVote],
) -> str:
    """Format a Telegram alert string (< 300 chars).

    Uses emoji prefixes based on trigger type and includes VIX level,
    council consensus, and dissent summary if present.
    """
    # Determine emoji — for OVERRIDE, use green/red based on state
    trigger = snapshot.trigger_type
    if trigger == "OVERRIDE_STATE_CHANGE" and snapshot.override:
        state = (snapshot.override.state or "").upper()
        if state in ("ON", "ON_CONFIRMED"):
            emoji = "\U0001f7e2"  # green
        else:
            emoji = "\U0001f534"  # red
    else:
        emoji = _TRIGGER_EMOJI.get(trigger, "\U0001f4ac")

    # VIX level
    vix_str = ""
    if snapshot.override and snapshot.override.vix is not None:
        vix_str = f" | VIX {snapshot.override.vix:.1f}"

    # Build alert
    trigger_short = trigger.replace("_", " ").title()
    parts = [
        f"{emoji} {trigger_short}",
        f"{result.decision} ({result.consensus_strength}, conf {result.confidence:.0%}){vix_str}",
    ]

    if result.dissent_summary:
        # Truncate dissent to fit 300 char limit
        max_dissent = 100
        dissent = result.dissent_summary
        if len(dissent) > max_dissent:
            dissent = dissent[:max_dissent] + "..."
        parts.append(f"Dissent: {dissent}")

    alert = "\n".join(parts)

    # Hard cap at 300 chars
    if len(alert) > 300:
        alert = alert[:297] + "..."

    return alert


# ── 5. Main Orchestrator ─────────────────────────────────────────────────────

async def analyze_trading(snapshot: MarketSnapshot) -> TradingDecision:
    """Full trading council pipeline.

    1. Build prompts from snapshot
    2. Call all council models in parallel
    3. Parse responses into ModelVotes
    4. Optionally run Stage 2 peer ranking (skipped for time-sensitive triggers)
    5. Aggregate into consensus
    6. Format Telegram alert
    7. Return TradingDecision with metadata
    """
    start_time = time.perf_counter()

    # 1. Build prompts
    system_prompt, messages = build_prompts(snapshot)

    # 2. Call models in parallel
    configs = [entry["config"] for entry in TRADING_MODELS.values()]
    model_names = list(TRADING_MODELS.keys())

    raw_responses = await call_models_parallel(
        configs, messages, timeout=TRADING_TIMEOUT_SECONDS
    )

    # Map ModelConfig → model_name for results
    config_to_name: dict[ModelConfig, str] = {}
    for name, entry in TRADING_MODELS.items():
        config_to_name[entry["config"]] = name

    # 3. Parse responses
    votes: dict[str, ModelVote] = {}
    for config, raw_text in raw_responses.items():
        name = config_to_name.get(config, config.display_name)
        votes[name] = parse_trading_response(raw_text, name)

    # 4. Stage 2 peer ranking (simplified — adjust confidence based on peer agreement)
    if snapshot.trigger_type not in SKIP_STAGE2_TRIGGERS:
        logger.debug("Running Stage 2 peer ranking for %s", snapshot.trigger_type)
        votes = await _stage2_peer_adjustment(votes, messages)

    # 5. Aggregate
    consensus = aggregate_decisions(votes)

    # 6. Format alert
    alert_text = format_telegram_alert(snapshot, consensus, votes)

    # 7. Build response
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Estimate token usage and cost (rough: ~4 chars per token)
    total_input_chars = sum(len(m.get("content", "")) for m in messages) * len(configs)
    total_output_chars = sum(
        len(raw or "") for raw in raw_responses.values()
    )
    est_input_tokens = total_input_chars // 4
    est_output_tokens = total_output_chars // 4
    est_cost = _estimate_cost(est_input_tokens, est_output_tokens)

    # Build action items from votes
    action_items: list[str] = []
    if consensus.decision in ("BUY", "CAUTIOUS_LONG"):
        action_items.append("Consider opening or adding to long positions")
    elif consensus.decision in ("SELL", "CAUTIOUS_SHORT"):
        action_items.append("Consider opening or adding to short positions")
    elif consensus.decision == "EXIT":
        action_items.append("Close all open positions")
    elif consensus.decision == "REDUCE":
        action_items.append("Reduce position sizing by 30-50%")

    if consensus.dissent_summary:
        action_items.append("Review dissenting analysis before acting")

    return TradingDecision(
        timestamp=snapshot.timestamp,
        trigger_type=snapshot.trigger_type,
        decision=consensus.decision,
        confidence=consensus.confidence,
        reasoning=_build_reasoning(consensus, votes),
        council_votes=votes,
        consensus=consensus.consensus,
        consensus_strength=consensus.consensus_strength,
        dissent_summary=consensus.dissent_summary,
        action_items=action_items,
        alert_text=alert_text,
        meta=TradingMeta(
            total_tokens=est_input_tokens + est_output_tokens,
            cost_usd=round(est_cost, 6),
            latency_ms=elapsed_ms,
            models_used=list(votes.keys()),
        ),
    )


# ── Internal Helpers ──────────────────────────────────────────────────────────

async def _stage2_peer_adjustment(
    votes: dict[str, ModelVote],
    original_messages: list[dict],
) -> dict[str, ModelVote]:
    """Simplified Stage 2: adjust confidence based on peer agreement.

    Models that agree with the majority get a small confidence boost;
    dissenters get a slight reduction. This is a lightweight alternative
    to full peer ranking for the trading use case.
    """
    if len(votes) < 2:
        return votes

    # Find majority decision
    decision_counts: Counter[str] = Counter(v.decision for v in votes.values())
    majority_decision = decision_counts.most_common(1)[0][0]

    adjusted = {}
    for name, vote in votes.items():
        if vote.decision == majority_decision:
            # Boost confidence slightly for agreement
            new_conf = min(1.0, vote.confidence * 1.05)
        else:
            # Reduce confidence slightly for dissent
            new_conf = max(0.0, vote.confidence * 0.95)

        adjusted[name] = ModelVote(
            decision=vote.decision,
            confidence=round(new_conf, 3),
            key_factor=vote.key_factor,
            risk_flag=vote.risk_flag,
            reasoning=vote.reasoning,
        )

    return adjusted


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost across all 3 models.

    Approximate rates per million tokens:
      Claude: $3 input, $15 output
      GPT:    $2 input, $8 output
      Gemini: $1.25 input, $10 output
    Per-model share is roughly 1/3 of total tokens each.
    """
    per_model_input = input_tokens / 3
    per_model_output = output_tokens / 3

    cost = 0.0
    # Claude
    cost += (per_model_input * 3.0 + per_model_output * 15.0) / 1_000_000
    # GPT
    cost += (per_model_input * 2.0 + per_model_output * 8.0) / 1_000_000
    # Gemini
    cost += (per_model_input * 1.25 + per_model_output * 10.0) / 1_000_000

    return cost


def _build_reasoning(
    consensus: ConsensusResult,
    votes: dict[str, ModelVote],
) -> str:
    """Build a combined reasoning string from the consensus and votes."""
    parts = [f"Council {consensus.consensus_strength} consensus: {consensus.decision}."]

    for name, vote in votes.items():
        if vote.reasoning:
            parts.append(f"{name}: {vote.reasoning}")

    return " ".join(parts)
