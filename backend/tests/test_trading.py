"""Tests for the Trading Council feature.

Covers parsing, consensus aggregation, pipeline logic, models, and alerts.
All LLM calls are mocked — no real API calls are made.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.llm_providers import ModelConfig
from backend.trading_config import TRADING_MODELS
from backend.trading_council import (
    aggregate_decisions,
    analyze_trading,
    format_telegram_alert,
    parse_trading_response,
)
from backend.trading_models import (
    ConsensusResult,
    GeoStressData,
    MarketSnapshot,
    ModelVote,
    OverrideData,
    TradingDecision,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_json_response() -> str:
    return json.dumps({
        "decision": "HOLD",
        "confidence": 0.7,
        "key_factor": "VIX elevated",
        "risk_flag": None,
        "reasoning": "VIX is above 25, suggesting caution",
    })


@pytest.fixture
def markdown_json_response() -> str:
    return '```json\n{"decision":"BUY","confidence":0.8,"key_factor":"Override ON","risk_flag":null,"reasoning":"Strong trend confirmed"}\n```'


@pytest.fixture
def override_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-03-11T16:45:00+02:00",
        trigger_type="OVERRIDE_STATE_CHANGE",
        trigger_details="Override ON -> OFF_WARNING",
        override=OverrideData(state="ON", previous_state="OFF", vix=25.15),
    )


@pytest.fixture
def geostress_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-03-11T16:45:00+02:00",
        trigger_type="GEOSTRESS_ALERT",
        geostress=GeoStressData(active=True, score=5.2),
    )


@pytest.fixture
def morning_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        timestamp="2026-03-11T09:00:00+02:00",
        trigger_type="MORNING_BRIEFING",
    )


@pytest.fixture
def unanimous_votes() -> dict[str, ModelVote]:
    return {
        "claude": ModelVote(decision="HOLD", confidence=0.7, key_factor="VIX high"),
        "gpt": ModelVote(decision="HOLD", confidence=0.6, key_factor="Caution"),
        "gemini": ModelVote(decision="HOLD", confidence=0.65, key_factor="Neutral"),
    }


@pytest.fixture
def majority_votes() -> dict[str, ModelVote]:
    return {
        "claude": ModelVote(decision="HOLD", confidence=0.7, key_factor="VIX"),
        "gpt": ModelVote(decision="HOLD", confidence=0.6, key_factor="EMA"),
        "gemini": ModelVote(decision="BUY", confidence=0.8, key_factor="Momentum"),
    }


@pytest.fixture
def split_votes() -> dict[str, ModelVote]:
    return {
        "claude": ModelVote(decision="HOLD", confidence=0.5, key_factor="VIX"),
        "gpt": ModelVote(decision="BUY", confidence=0.6, key_factor="Override"),
        "gemini": ModelVote(decision="SELL", confidence=0.4, key_factor="GeoStress"),
    }


def _mock_parallel_response(response_json: str) -> dict[ModelConfig, str]:
    """Build a mock return value for call_models_parallel keyed by ModelConfig."""
    return {
        entry["config"]: response_json
        for entry in TRADING_MODELS.values()
    }


# ── PARSING TESTS ─────────────────────────────────────────────────────────────

class TestParsing:
    def test_parse_valid_json_response(self, valid_json_response: str) -> None:
        vote = parse_trading_response(valid_json_response, "test_model")
        assert vote.decision == "HOLD"
        assert vote.confidence == 0.7
        assert vote.key_factor == "VIX elevated"
        assert vote.reasoning == "VIX is above 25, suggesting caution"

    def test_parse_json_in_markdown(self, markdown_json_response: str) -> None:
        vote = parse_trading_response(markdown_json_response, "test_model")
        assert vote.decision == "BUY"
        assert vote.confidence == 0.8
        assert vote.key_factor == "Override ON"

    def test_parse_malformed_response(self) -> None:
        vote = parse_trading_response("garbage text with no JSON", "test_model")
        assert vote.decision == "HOLD"
        assert vote.confidence == 0.3
        assert vote.key_factor == "Parse failed"
        assert vote.risk_flag == "Raw response could not be parsed"

    def test_parse_partial_json(self) -> None:
        partial = json.dumps({"decision": "EXIT", "confidence": 0.9})
        vote = parse_trading_response(partial, "test_model")
        assert vote.decision == "EXIT"
        assert vote.confidence == 0.9
        assert vote.key_factor == ""  # default
        assert vote.reasoning == ""  # default

    def test_parse_with_extra_text(self) -> None:
        text = 'Here is my analysis: {"decision": "SELL", "confidence": 0.75, "key_factor": "bearish", "risk_flag": null, "reasoning": "downtrend"}'
        vote = parse_trading_response(text, "test_model")
        # Tier 1 fails (extra text), Tier 2 fails (no code block),
        # Tier 3 regex should extract fields
        assert vote.decision == "SELL"
        assert vote.confidence == 0.75


# ── CONSENSUS TESTS ───────────────────────────────────────────────────────────

class TestConsensus:
    def test_consensus_unanimous(self, unanimous_votes: dict[str, ModelVote]) -> None:
        result = aggregate_decisions(unanimous_votes)
        assert result.decision == "HOLD"
        assert result.consensus_strength == "UNANIMOUS"
        assert result.dissent_summary == ""
        assert result.confidence > 0.5

    def test_consensus_majority(self, majority_votes: dict[str, ModelVote]) -> None:
        result = aggregate_decisions(majority_votes)
        assert result.decision == "HOLD"
        assert result.consensus_strength == "2/3"
        assert "gemini" in result.dissent_summary
        assert "BUY" in result.dissent_summary

    def test_consensus_three_way_split(self, split_votes: dict[str, ModelVote]) -> None:
        result = aggregate_decisions(split_votes)
        assert result.decision == "HOLD"  # fallback on 3-way split
        assert result.consensus_strength == "1/3"

    def test_consensus_weighted_confidence(self) -> None:
        votes = {
            "claude": ModelVote(decision="HOLD", confidence=1.0, key_factor="a"),
            "gpt": ModelVote(decision="HOLD", confidence=1.0, key_factor="b"),
            "gemini": ModelVote(decision="HOLD", confidence=1.0, key_factor="c"),
        }
        result = aggregate_decisions(votes)
        # Weights: claude=0.40, gpt=0.35, gemini=0.25, sum=1.0
        # Weighted confidence = (1.0*0.40 + 1.0*0.35 + 1.0*0.25) / 1.0 * 1.0 = 1.0
        assert abs(result.confidence - 1.0) < 0.01


# ── PIPELINE TESTS ────────────────────────────────────────────────────────────

class TestPipeline:
    @pytest.mark.asyncio
    @patch("backend.trading_council.call_models_parallel", new_callable=AsyncMock)
    async def test_skip_stage2_for_override(
        self, mock_parallel: AsyncMock, override_snapshot: MarketSnapshot
    ) -> None:
        mock_response = json.dumps({
            "decision": "HOLD", "confidence": 0.6,
            "key_factor": "test", "risk_flag": None, "reasoning": "test",
        })
        mock_parallel.return_value = _mock_parallel_response(mock_response)

        result = await analyze_trading(override_snapshot)

        assert result.decision == "HOLD"
        # call_models_parallel should be called exactly once (Stage 1 only)
        assert mock_parallel.call_count == 1

    @pytest.mark.asyncio
    @patch("backend.trading_council.call_models_parallel", new_callable=AsyncMock)
    async def test_skip_stage2_for_geostress(
        self, mock_parallel: AsyncMock, geostress_snapshot: MarketSnapshot
    ) -> None:
        mock_response = json.dumps({
            "decision": "EXIT", "confidence": 0.85,
            "key_factor": "geo", "risk_flag": "high stress", "reasoning": "risk off",
        })
        mock_parallel.return_value = _mock_parallel_response(mock_response)

        result = await analyze_trading(geostress_snapshot)

        assert result.decision == "EXIT"
        assert mock_parallel.call_count == 1

    @pytest.mark.asyncio
    @patch("backend.trading_council.call_models_parallel", new_callable=AsyncMock)
    async def test_morning_briefing_full_pipeline(
        self, mock_parallel: AsyncMock, morning_snapshot: MarketSnapshot
    ) -> None:
        mock_response = json.dumps({
            "decision": "HOLD", "confidence": 0.5,
            "key_factor": "morning", "risk_flag": None, "reasoning": "pre-market",
        })
        mock_parallel.return_value = _mock_parallel_response(mock_response)

        result = await analyze_trading(morning_snapshot)

        assert result.decision == "HOLD"
        # MORNING_BRIEFING is NOT in SKIP_STAGE2_TRIGGERS, so stage2
        # peer adjustment runs (but it doesn't call call_models_parallel again
        # in the simplified implementation — it just adjusts confidence)
        assert mock_parallel.call_count == 1
        assert result.trigger_type == "MORNING_BRIEFING"


# ── MODEL TESTS ───────────────────────────────────────────────────────────────

class TestModels:
    def test_market_snapshot_minimal(self) -> None:
        snap = MarketSnapshot(
            timestamp="2026-03-11T16:45:00+02:00",
            trigger_type="GEOSTRESS_ALERT",
        )
        assert snap.trigger_type == "GEOSTRESS_ALERT"
        assert snap.override is None
        assert snap.geostress is None

    def test_market_snapshot_full(self) -> None:
        snap = MarketSnapshot(
            timestamp="2026-03-11T16:45:00+02:00",
            trigger_type="OVERRIDE_STATE_CHANGE",
            trigger_details="Override ON -> OFF_WARNING",
            override=OverrideData(
                state="OFF_WARNING", previous_state="ON",
                z15=-0.45, z30=-0.82, vix=25.15, rebound_pct=8.2,
            ),
            geostress=GeoStressData(active=False, score=1.2),
        )
        assert snap.override.state == "OFF_WARNING"
        assert snap.override.vix == 25.15
        assert snap.geostress.active is False

    def test_trading_decision_serialization(self) -> None:
        decision = TradingDecision(
            timestamp="2026-03-11T16:45:00+02:00",
            trigger_type="OVERRIDE_STATE_CHANGE",
            decision="HOLD",
            confidence=0.65,
            reasoning="Test reasoning",
            council_votes={
                "claude": ModelVote(decision="HOLD", confidence=0.7, key_factor="vix"),
            },
            consensus="UNANIMOUS_HOLD",
            consensus_strength="UNANIMOUS",
        )
        # Round-trip: model -> JSON -> model
        json_str = decision.model_dump_json()
        restored = TradingDecision.model_validate_json(json_str)
        assert restored.decision == "HOLD"
        assert restored.confidence == 0.65
        assert "claude" in restored.council_votes
        assert restored.council_votes["claude"].decision == "HOLD"


# ── ALERT TESTS ───────────────────────────────────────────────────────────────

class TestAlerts:
    def test_alert_override_on(self, override_snapshot: MarketSnapshot) -> None:
        result = ConsensusResult(
            decision="HOLD", confidence=0.65,
            consensus="UNANIMOUS_HOLD", consensus_strength="UNANIMOUS",
        )
        votes: dict[str, ModelVote] = {}
        alert = format_telegram_alert(override_snapshot, result, votes)
        assert "\U0001f7e2" in alert  # green circle for Override ON
        assert "VIX 25.1" in alert or "VIX 25.2" in alert

    def test_alert_geostress(self, geostress_snapshot: MarketSnapshot) -> None:
        result = ConsensusResult(
            decision="EXIT", confidence=0.85,
            consensus="UNANIMOUS_EXIT", consensus_strength="UNANIMOUS",
        )
        votes: dict[str, ModelVote] = {}
        alert = format_telegram_alert(geostress_snapshot, result, votes)
        assert "\U0001f6a8" in alert  # rotating light for geostress
        assert "EXIT" in alert

    def test_alert_length(
        self,
        override_snapshot: MarketSnapshot,
        geostress_snapshot: MarketSnapshot,
        morning_snapshot: MarketSnapshot,
    ) -> None:
        result = ConsensusResult(
            decision="HOLD", confidence=0.65,
            consensus="2/3_HOLD", consensus_strength="2/3",
            dissent_summary="gemini suggests BUY based on strong momentum and positive cross-asset signals across multiple indicators",
        )
        votes: dict[str, ModelVote] = {}

        for snap in [override_snapshot, geostress_snapshot, morning_snapshot]:
            alert = format_telegram_alert(snap, result, votes)
            assert len(alert) <= 300, f"Alert too long ({len(alert)} chars): {alert}"
