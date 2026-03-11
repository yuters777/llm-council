"""Pydantic v2 models for the Trading Council feature.

Input models define the MarketSnapshot schema received from the Python trading engine.
Output models define the TradingDecision schema returned to the caller.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Input Models ──────────────────────────────────────────────────────────────

class OverrideData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: Optional[str] = None
    previous_state: Optional[str] = None
    z15: Optional[float] = None
    z30: Optional[float] = None
    override_score: Optional[float] = None
    vix: Optional[float] = None
    vix_5min_ago: Optional[float] = None
    vvix: Optional[float] = None
    rebound_pct: Optional[float] = None
    term_structure: Optional[float] = None
    vix_fatigue_modifier: Optional[float] = None


class GeoStressComponents(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    z_dvix_5: Optional[float] = None
    z_dvvix_5: Optional[float] = None
    z_gold_5: Optional[float] = None
    z_oil_5_abs: Optional[float] = None
    z_jpy_dxy: Optional[float] = None
    breadth_shock: Optional[float] = None


class GeoStressData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    active: Optional[bool] = None
    score: Optional[float] = None
    components: Optional[GeoStressComponents] = None


class TickerEmaState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state_4h: Optional[str] = Field(None, alias="4h")
    state_m5: Optional[str] = Field(None, alias="m5")
    score: Optional[float] = None


class EmaGateData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state_4h: Optional[str] = None
    state_m5: Optional[str] = None
    m5_substate: Optional[str] = None
    trend_score: Optional[float] = None
    adx: Optional[float] = None
    ticker_states: Optional[dict[str, TickerEmaState]] = None


class CryptoOverrideData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: Optional[str] = None
    dvol: Optional[float] = None
    ethdvol: Optional[float] = None
    btc_funding_rate: Optional[float] = None
    hierarchical_weight: Optional[float] = None


class AssetPrice(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    price: Optional[float] = None
    change_1h_pct: Optional[float] = None
    volume_vs_median: Optional[float] = None


class CrossAssetData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    btc: Optional[AssetPrice] = None
    eth: Optional[AssetPrice] = None
    oil_wti: Optional[float] = None
    oil_change_30m_sigma: Optional[float] = None
    gold: Optional[float] = None
    coin_leading: Optional[bool] = None
    ibit_vs_btc_divergence: Optional[float] = None
    china_adrs: Optional[float] = None


class SessionData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    zone: Optional[str] = None
    zone_reliability: Optional[str] = None
    minutes_to_close: Optional[int] = None
    is_event_day: Optional[bool] = None
    event_quarantine: Optional[bool] = None
    day_of_week: Optional[str] = None


class RecentAlert(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    time: Optional[str] = None
    type: Optional[str] = None
    details: Optional[str] = None
    ticker: Optional[str] = None
    timeframe: Optional[str] = None


TRIGGER_TYPES = Literal[
    "OVERRIDE_STATE_CHANGE",
    "GEOSTRESS_ALERT",
    "CONFLICTING_SIGNALS",
    "MORNING_BRIEFING",
    "UNUSUAL_PATTERN",
    "EARNINGS_PROXIMITY",
]


class MarketSnapshot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str
    trigger_type: TRIGGER_TYPES
    trigger_details: Optional[str] = None
    override: Optional[OverrideData] = None
    geostress: Optional[GeoStressData] = None
    ema_gate: Optional[EmaGateData] = None
    crypto_override: Optional[CryptoOverrideData] = None
    cross_asset: Optional[CrossAssetData] = None
    session: Optional[SessionData] = None
    recent_alerts: Optional[list[RecentAlert]] = None


# ── Output Models ─────────────────────────────────────────────────────────────

DECISION_TYPES = Literal[
    "BUY", "SELL", "HOLD", "EXIT", "REDUCE", "CAUTIOUS_SHORT", "CAUTIOUS_LONG"
]


class ModelVote(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: DECISION_TYPES = "HOLD"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_factor: str = ""
    risk_flag: Optional[str] = None
    reasoning: str = ""


class ConsensusResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: str = "HOLD"
    confidence: float = 0.5
    consensus: str = ""
    consensus_strength: str = ""
    dissent_summary: str = ""


class TradingMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    models_used: list[str] = Field(default_factory=list)


class TradingDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: str = ""
    trigger_type: str = ""
    decision: str = "HOLD"
    confidence: float = 0.5
    reasoning: str = ""
    council_votes: dict[str, ModelVote] = Field(default_factory=dict)
    consensus: str = ""
    consensus_strength: str = ""
    dissent_summary: str = ""
    action_items: list[str] = Field(default_factory=list)
    alert_text: str = ""
    meta: TradingMeta = Field(default_factory=TradingMeta)
