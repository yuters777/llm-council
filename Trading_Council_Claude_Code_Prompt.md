# Claude Code Implementation Plan: Trading Council Mode

**Project:** yuters777/llm-council (GitHub)
**Date:** March 11, 2026
**Spec:** Trading_Council_TZ_for_LLM_Council.md

---

## NOTES FOR CLAUDE CODE

**Architecture rule:** The backend uses direct API calls to OpenAI, Anthropic, Google — via `backend/llm_providers.py` which exposes `ModelConfig`, `call_model()`, `call_models_parallel()`. If `llm_providers.py` does not exist yet, Step 0 creates it. All provider calls go through this layer — never call APIs directly from business logic.

**Non-negotiable constraints:**
- Existing web UI and `/api/conversations/*` routes must NOT change
- All network I/O: `async` + `httpx.AsyncClient`
- Type hints on every function signature
- No secrets in logs (API keys, auth headers, full request bodies)
- Config via `.env` variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- Python style: small composable functions, Pydantic models for all JSON contracts
- Backend managed by `uv`, runs on port 8001

**How to run:**
```bash
uv run python -m backend.main       # backend
cd frontend && npm run dev           # frontend
```

---

## Summary Table

| Step | Scope | Files Created/Modified | Depends On |
|------|-------|----------------------|------------|
| 0 | Provider abstraction layer | `backend/llm_providers.py`, `backend/config.py` | — |
| 1 | Pydantic models + trading config | `backend/trading_models.py`, `backend/trading_config.py` | Step 0 |
| 2 | Prompt templates | `backend/prompts/trading/*.md` (6 files) | — |
| 3 | Core trading council logic | `backend/trading_council.py` | Steps 0, 1, 2 |
| 4 | FastAPI endpoint | `backend/main.py` (add route) | Step 3 |
| 5 | Unit tests | `backend/tests/test_trading.py` | Steps 1, 3 |
| 6 | Docs + integration test | `CLAUDE.md`, `README.md` | All above |

---

## STEP 0: Ensure direct-API provider layer exists

**Goal:** Verify or create `backend/llm_providers.py` — the canonical abstraction for calling OpenAI, Anthropic, and Google Gemini directly.

**Context for Claude Code:** Read `backend/config.py`, `backend/openrouter.py`, `backend/council.py`

### Prompt:

```
Check if `backend/llm_providers.py` exists with `ModelConfig`, `call_model()`, and `call_models_parallel()`.

IF IT EXISTS and already calls OpenAI/Anthropic/Google directly:
- Read it, confirm the interface matches what `backend/council.py` expects
- Skip to verification

IF IT DOES NOT EXIST (repo still uses openrouter.py):
Create `backend/llm_providers.py` with:

1. ModelConfig dataclass/Pydantic model:
   - provider: Literal["openai", "anthropic", "google"]
   - model: str
   - temperature: float = 0.7
   - max_tokens: int = 4096
   - timeout: float = 120.0

2. async def call_model(config: ModelConfig, messages: list[dict], system_prompt: str | None = None) -> dict | None:
   - Routes to provider-specific implementation
   - Returns {"content": str, "usage": {...}} or None on failure
   - OpenAI: httpx POST to https://api.openai.com/v1/chat/completions
   - Anthropic: httpx POST to https://api.anthropic.com/v1/messages (system goes in top-level "system" field, NOT in messages)
   - Google: httpx POST to https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
   - Each provider function: proper headers, API key from env, structured error handling
   - Log errors with model name but NOT the API key or full request body

3. async def call_models_parallel(configs: list[ModelConfig], messages: list[dict], system_prompt: str | None = None) -> dict[str, dict | None]:
   - asyncio.gather() over call_model() for each config
   - Returns {model_name: response_dict}
   - Graceful degradation: failed models return None, don't crash the batch

4. Update `backend/config.py`:
   - Add OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY from env
   - Keep existing COUNCIL_MODELS / CHAIRMAN_MODEL / OPENROUTER_API_KEY (don't break existing flow)

CRITICAL: Do NOT remove or modify `backend/openrouter.py` or existing council.py logic.
The new llm_providers.py is an ADDITIONAL module — the trading council will use it,
while the existing chat flow continues to use openrouter.py until separately migrated.

CRITICAL: Anthropic API — system prompt goes in the top-level "system" field, NOT as a message with role "system".

CRITICAL: Google Gemini API — uses "contents" not "messages", role is "user"/"model" not "user"/"assistant".

VERIFY:
- `uv run python -c "from backend.llm_providers import ModelConfig, call_model, call_models_parallel; print('OK')"`
- All three env vars loadable (warn if missing, don't crash)
```

### Verification Criteria:
- [ ] `ModelConfig` importable with all fields
- [ ] `call_model()` handles all 3 providers with correct API formats
- [ ] `call_models_parallel()` returns dict keyed by model name
- [ ] No existing files broken — `backend/openrouter.py` and `backend/council.py` unchanged
- [ ] API keys read from env, not hardcoded

---

## STEP 1: Pydantic models + trading config

**Goal:** Define the full MarketSnapshot input schema, TradingDecision output schema, and trading-specific configuration.

**Context for Claude Code:** `Trading_Council_TZ_for_LLM_Council.md` §3.2, §3.3, §7; `backend/llm_providers.py`

### Prompt:

```
Create two new files for the Trading Council feature.

FILE 1: `backend/trading_models.py`

Define Pydantic v2 models matching the spec exactly. All nested objects are separate models.

Input models (all fields Optional except timestamp and trigger_type — the Python engine
may send minimal snapshots):

- OverrideData: state, previous_state, z15, z30, override_score, vix, vix_5min_ago, vvix, rebound_pct, term_structure, vix_fatigue_modifier
- GeoStressComponents: z_dvix_5, z_dvvix_5, z_gold_5, z_oil_5_abs, z_jpy_dxy, breadth_shock
- GeoStressData: active, score, components (GeoStressComponents | None)
- TickerEmaState: 4h → field alias "state_4h", m5 → "state_m5", score
- EmaGateData: state_4h, state_m5, m5_substate, trend_score, adx, ticker_states (dict[str, TickerEmaState] | None)
- CryptoOverrideData: state, dvol, ethdvol, btc_funding_rate, hierarchical_weight
- AssetPrice: price, change_1h_pct, volume_vs_median
- CrossAssetData: btc, eth (AssetPrice | None), oil_wti, oil_change_30m_sigma, gold, coin_leading, ibit_vs_btc_divergence, china_adrs
- SessionData: zone, zone_reliability, minutes_to_close, is_event_day, event_quarantine, day_of_week
- RecentAlert: time, type, details, ticker (optional), timeframe (optional)
- MarketSnapshot: timestamp, trigger_type, trigger_details, override, geostress, ema_gate, crypto_override, cross_asset, session, recent_alerts (list[RecentAlert] | None)
  - trigger_type: Literal["OVERRIDE_STATE_CHANGE", "GEOSTRESS_ALERT", "CONFLICTING_SIGNALS", "MORNING_BRIEFING", "UNUSUAL_PATTERN", "EARNINGS_PROXIMITY"]

Output models:
- ModelVote: decision (Literal["BUY","SELL","HOLD","EXIT","REDUCE","CAUTIOUS_SHORT","CAUTIOUS_LONG"]), confidence (float 0-1), key_factor (str), risk_flag (str | None), reasoning (str)
- ConsensusResult: decision, confidence, consensus (str), consensus_strength (str), dissent_summary (str)
- TradingMeta: total_tokens (int), cost_usd (float), latency_ms (int), models_used (list[str])
- TradingDecision: timestamp, trigger_type, decision, confidence, reasoning, council_votes (dict[str, ModelVote]), consensus, consensus_strength, dissent_summary, action_items (list[str]), alert_text (str), meta (TradingMeta)

Use model_config = ConfigDict(populate_by_name=True) where field aliases are needed.

FILE 2: `backend/trading_config.py`

```python
from backend.llm_providers import ModelConfig

TRADING_ENABLED: bool = True  # from env TRADING_COUNCIL_ENABLED, default True

TRADING_MODELS: dict[str, dict] = {
    "claude": {"weight": 0.40, "config": ModelConfig(provider="anthropic", model="claude-sonnet-4-20250514", max_tokens=1000, temperature=0.3)},
    "gpt": {"weight": 0.35, "config": ModelConfig(provider="openai", model="gpt-4.1", max_tokens=1000, temperature=0.3)},
    "gemini": {"weight": 0.25, "config": ModelConfig(provider="google", model="gemini-3-pro-preview", max_tokens=1000, temperature=0.3)},
}

SKIP_STAGE2_TRIGGERS: set[str] = {"OVERRIDE_STATE_CHANGE", "GEOSTRESS_ALERT", "UNUSUAL_PATTERN"}

TRADING_TIMEOUT_SECONDS: float = 30.0
FALLBACK_DECISION: str = "HOLD"
FALLBACK_CONFIDENCE: float = 0.3
PROMPTS_DIR: str = "backend/prompts/trading"
```

Temperature 0.3 (not 0.7) — we want deterministic trading analysis, not creative writing.

CRITICAL: Keep `Optional` on all MarketSnapshot nested fields. The Python engine will often
send partial snapshots. Pydantic must not reject them.

VERIFY:
- `uv run python -c "from backend.trading_models import MarketSnapshot, TradingDecision; print('OK')"`
- `uv run python -c "from backend.trading_config import TRADING_MODELS; print(list(TRADING_MODELS.keys()))"`
- Validate the sample curl JSON from spec §11 parses into MarketSnapshot without errors
```

### Verification Criteria:
- [ ] `MarketSnapshot(**sample_json)` succeeds with the minimal curl payload from spec §11
- [ ] `TradingDecision` model has all fields from spec §3.3
- [ ] `TRADING_MODELS` contains 3 entries with correct weights summing to 1.0
- [ ] All models use type hints, no `Any` unless truly needed

---

## STEP 2: Prompt templates

**Goal:** Create 6 markdown prompt template files, loadable at runtime.

**Context for Claude Code:** `Trading_Council_TZ_for_LLM_Council.md` §5.1-5.4

### Prompt:

```
Create directory `backend/prompts/trading/` with 6 markdown files.

Each file is a Jinja2-style template (but we'll use simple str.format() / .replace()).
Placeholders: {snapshot_json}, {news_json}, {movers_json}

FILE 1: `system_base.md`
Copy the base system prompt from spec §5.1 EXACTLY, including the JSON response format block.
This is the system prompt sent to ALL models for ALL trigger types.

FILE 2: `override_change.md`
Copy spec §5.2 — the user-message template for OVERRIDE_STATE_CHANGE triggers.
Must contain {snapshot_json} placeholder.

FILE 3: `geostress_alert.md`
Copy spec §5.3 — for GEOSTRESS_ALERT triggers.
Must contain {snapshot_json} placeholder.

FILE 4: `morning_briefing.md`
Copy spec §5.4 — for MORNING_BRIEFING triggers.
Must contain {snapshot_json}, {news_json}, {movers_json} placeholders.

FILE 5: `conflicting_signals.md`
Create a prompt for CONFLICTING_SIGNALS trigger (not fully specified in the TZ, design it):
- Explain that Override and EMA states are sending contradictory signals
- Ask: which signal takes priority per framework hierarchy?
- Ask: recommended position sizing given the conflict?
- Ask: what would resolve the conflict?
Must contain {snapshot_json} placeholder.

FILE 6: `unusual_pattern.md`
Create a prompt for UNUSUAL_PATTERN trigger (not fully specified, design it):
- Explain that an anomalous pattern was detected (TWAP, volume anomaly, COIN divergence, etc.)
- Ask: is this pattern actionable or noise?
- Ask: what's the expected duration/impact?
- Ask: recommended response given current Override + EMA state?
Must contain {snapshot_json} placeholder.

CRITICAL: The system_base.md prompt must end with the exact JSON response format
from the spec (decision, confidence, key_factor, risk_flag, reasoning).
Models MUST know what format to respond in.

VERIFY:
- All 6 files exist in backend/prompts/trading/
- Each contains the appropriate {snapshot_json} placeholder
- morning_briefing.md additionally contains {news_json} and {movers_json}
- system_base.md contains the JSON response format block
```

### Verification Criteria:
- [ ] 6 `.md` files in `backend/prompts/trading/`
- [ ] system_base.md contains full framework rules + JSON format
- [ ] All user-message templates contain `{snapshot_json}`
- [ ] morning_briefing.md has all 3 placeholders

---

## STEP 3: Core trading council logic

**Goal:** Implement `backend/trading_council.py` — prompt building, response parsing, consensus aggregation, alert formatting.

**Context for Claude Code:** `backend/llm_providers.py`, `backend/trading_models.py`, `backend/trading_config.py`, `backend/prompts/trading/*.md`, spec §6.1-6.4

### Prompt:

```
Create `backend/trading_council.py` with the following functions.
This is the core business logic. It calls models ONLY through llm_providers.call_models_parallel().

1. PROMPT BUILDING:

def load_prompt_template(trigger_type: str) -> str:
    """Load the user-message template for a given trigger type from prompts/trading/.
    Cache templates in a module-level dict after first load.
    Map trigger_type → filename: OVERRIDE_STATE_CHANGE → override_change.md, etc.
    Raise ValueError for unknown trigger types."""

def load_system_prompt() -> str:
    """Load and cache system_base.md."""

def build_prompts(snapshot: MarketSnapshot) -> tuple[str, list[dict]]:
    """Build system prompt + user messages for the council call.
    Returns (system_prompt, messages) where messages = [{"role": "user", "content": ...}].
    For MORNING_BRIEFING: include news_json and movers_json placeholders (empty dict if not provided).
    For all others: inject snapshot.model_dump_json(indent=2) into {snapshot_json}."""

2. RESPONSE PARSING (3-tier fallback per spec §6.2):

def parse_trading_response(raw_text: str, model_name: str) -> ModelVote:
    """Parse a single model's raw text into ModelVote.
    Try 1: json.loads(raw_text.strip())
    Try 2: extract JSON from ```json ... ``` markdown block
    Try 3: regex extract "decision": "...", "confidence": ..., "reasoning": "..."
    Fallback: ModelVote(decision=FALLBACK_DECISION, confidence=FALLBACK_CONFIDENCE,
              key_factor="Parse failed", risk_flag="Raw response could not be parsed",
              reasoning=raw_text[:500])
    Log which tier succeeded for debugging (but don't log full raw_text in production)."""

3. CONSENSUS AGGREGATION (per spec §6.3):

def aggregate_decisions(votes: dict[str, ModelVote]) -> ConsensusResult:
    """Aggregate model votes into consensus.
    - Count votes per decision type
    - Majority vote wins; on 3-way split → HOLD
    - Weighted confidence: sum(vote.confidence * model_weight)
      Weights from TRADING_MODELS config: claude=0.40, gpt=0.35, gemini=0.25
    - Identify dissenters (models that disagree with majority)
    - consensus_strength: "UNANIMOUS" or "2/3" or "1/3"
    - Format dissent_summary: "{model} suggests {decision} based on {key_factor}"
    """

4. ALERT FORMATTING (per spec §6.4):

def format_telegram_alert(snapshot: MarketSnapshot, result: ConsensusResult, votes: dict[str, ModelVote]) -> str:
    """Format Telegram alert string based on trigger_type.
    Use emoji prefixes per spec:
    🟢 OVERRIDE ON, 🔴 OVERRIDE OFF, 🚨 GEOSTRESS, ⚠️ CONFLICTING,
    📋 MORNING BRIEFING, 🔍 UNUSUAL PATTERN
    Include: VIX level, council consensus, dissent if any.
    Keep under 300 chars for Telegram readability."""

5. MAIN ORCHESTRATOR (per spec §6.1):

async def analyze_trading(snapshot: MarketSnapshot) -> TradingDecision:
    """Full trading council pipeline:
    1. Build prompts
    2. Call all 3 models in parallel via call_models_parallel()
       - Pass system_prompt separately (call_model supports it)
       - Use TRADING_MODELS configs
    3. Parse responses → ModelVote per model
    4. If trigger_type NOT in SKIP_STAGE2_TRIGGERS:
       - Run stage2 peer ranking (reuse logic from council.py's stage2_collect_rankings
         or implement a simplified version — models rank each other's trading analysis)
       - Apply ranking weights to confidence scores
    5. Aggregate decisions → ConsensusResult
    6. Format Telegram alert
    7. Build and return TradingDecision with meta (tokens, cost, latency)
    Track latency with time.perf_counter().
    Track tokens from call_model() response usage data.
    Estimate cost: claude=$3/MTok input + $15/MTok output,
                   gpt=$2/MTok input + $8/MTok output,
                   gemini=$1.25/MTok input + $10/MTok output (approximate)
    """

CRITICAL: All model calls go through `call_models_parallel()` from `llm_providers.py`.
Never import httpx or call APIs directly in this file.

CRITICAL: The parse function must be robust. Models WILL return broken JSON,
markdown-wrapped JSON, or plain text. The fallback to HOLD with low confidence
is a safety feature — it prevents the system from taking action on unparseable output.

CRITICAL: Use `import logging; logger = logging.getLogger(__name__)` for all logging.
Never print() in production code.

KEEP IDENTICAL: backend/council.py, backend/openrouter.py, backend/main.py

VERIFY:
- `uv run python -c "from backend.trading_council import analyze_trading, parse_trading_response; print('OK')"`
- parse_trading_response('{"decision":"HOLD","confidence":0.7,"key_factor":"test","risk_flag":null,"reasoning":"test"}', "test") returns ModelVote
- parse_trading_response('```json\n{"decision":"BUY","confidence":0.8,"key_factor":"x","risk_flag":null,"reasoning":"y"}\n```', "test") returns ModelVote with decision="BUY"
- parse_trading_response('garbage text', "test") returns ModelVote with decision="HOLD", confidence=0.3
```

### Verification Criteria:
- [ ] All 5 function groups implemented with proper type hints
- [ ] 3-tier JSON parse fallback works for clean JSON, markdown-wrapped JSON, and garbage
- [ ] Consensus aggregation handles unanimous, majority, and 3-way split
- [ ] Alert formatter produces readable Telegram strings < 300 chars
- [ ] `analyze_trading()` is async and calls only `call_models_parallel()`
- [ ] No direct API calls or httpx imports in this file

---

## STEP 4: FastAPI endpoint

**Goal:** Add `POST /api/trading/analyze` to `backend/main.py` — isolated from existing routes.

**Context for Claude Code:** `backend/main.py`, `backend/trading_models.py`, `backend/trading_council.py`, `backend/trading_config.py`

### Prompt:

```
Add the trading analysis endpoint to `backend/main.py`.

ADD these imports at the top:
from .trading_models import MarketSnapshot, TradingDecision
from .trading_council import analyze_trading
from .trading_config import TRADING_ENABLED

ADD this endpoint AFTER all existing /api/conversations/* routes:

@app.post("/api/trading/analyze", response_model=TradingDecision)
async def trading_analyze(snapshot: MarketSnapshot):
    """Analyze market snapshot through the Trading Council.
    
    Accepts a MarketSnapshot from the Python trading engine,
    runs it through 3 LLMs in parallel, and returns a consensus
    TradingDecision with individual votes and alert text.
    """
    if not TRADING_ENABLED:
        raise HTTPException(status_code=503, detail="Trading Council is disabled")
    
    try:
        result = await analyze_trading(snapshot)
        return result
    except Exception as e:
        logger.exception("Trading analysis failed")
        raise HTTPException(status_code=500, detail=f"Trading analysis failed: {str(e)}")

Also add a health-check extension to the existing root endpoint:

Modify the GET "/" handler to also return trading_enabled status:
    return {"status": "ok", "service": "LLM Council API", "trading_enabled": TRADING_ENABLED}

ADD `import logging` and `logger = logging.getLogger(__name__)` if not already present.

CRITICAL: Do NOT modify any existing endpoints. Do NOT change the conversation routes,
CORS settings, Pydantic models for conversations, or any imports related to council.py.

CRITICAL: The new endpoint is at /api/trading/analyze — NOT /api/conversations/trading.
It's a separate API surface for machine-to-machine calls from the Python engine.

KEEP IDENTICAL: All /api/conversations/* routes, CreateConversationRequest, SendMessageRequest,
ConversationMetadata, Conversation models, all imports from .council and .storage.

VERIFY:
- `uv run python -m backend.main` starts without errors
- `curl http://localhost:8001/` returns {"status":"ok","service":"LLM Council API","trading_enabled":true}
- `curl -X POST http://localhost:8001/api/trading/analyze -H "Content-Type: application/json" -d '{"timestamp":"2026-03-11T16:45:00+02:00","trigger_type":"OVERRIDE_STATE_CHANGE","trigger_details":"test"}' ` returns a TradingDecision JSON (or 500 if API keys not configured — that's OK for now)
- Existing `curl http://localhost:8001/api/conversations` still works
```

### Verification Criteria:
- [ ] Server starts without import errors
- [ ] `POST /api/trading/analyze` accepts MarketSnapshot and returns TradingDecision
- [ ] Existing `/api/conversations/*` routes unchanged and working
- [ ] Health check shows `trading_enabled` status
- [ ] Proper error handling with 503 (disabled) and 500 (failure)

---

## STEP 5: Unit tests

**Goal:** Comprehensive tests for parsing, consensus, and pipeline logic.

**Context for Claude Code:** `backend/trading_council.py`, `backend/trading_models.py`, `backend/trading_config.py`, spec §8.1

### Prompt:

```
Create `backend/tests/test_trading.py` with pytest tests.
Also create `backend/tests/__init__.py` if it doesn't exist.

Tests to write (per spec §8.1 + additional coverage):

PARSING TESTS:
1. test_parse_valid_json_response — clean JSON → correct ModelVote
2. test_parse_json_in_markdown — ```json ... ``` → extracted and parsed
3. test_parse_malformed_response — garbage → fallback HOLD, confidence=0.3
4. test_parse_partial_json — JSON missing some fields → fills defaults
5. test_parse_with_extra_text — "Here's my analysis: {...}" → extracts JSON

CONSENSUS TESTS:
6. test_consensus_unanimous — all 3 HOLD → UNANIMOUS_HOLD, high confidence
7. test_consensus_majority — 2 HOLD + 1 BUY → MAJORITY_HOLD, dissent captured
8. test_consensus_three_way_split — HOLD + BUY + SELL → HOLD fallback, low confidence
9. test_consensus_weighted_confidence — verify weights: claude=0.40, gpt=0.35, gemini=0.25

PIPELINE TESTS:
10. test_skip_stage2_for_override — OVERRIDE_STATE_CHANGE trigger skips Stage 2
11. test_skip_stage2_for_geostress — GEOSTRESS_ALERT trigger skips Stage 2
12. test_morning_briefing_full_pipeline — MORNING_BRIEFING does NOT skip Stage 2

MODEL TESTS:
13. test_market_snapshot_minimal — minimal payload (just timestamp + trigger_type) validates
14. test_market_snapshot_full — full payload from spec §3.2 validates
15. test_trading_decision_serialization — TradingDecision round-trips to/from JSON

ALERT TESTS:
16. test_alert_override_on — contains 🟢 and VIX value
17. test_alert_geostress — contains 🚨 and SUSPENDED
18. test_alert_length — all alerts < 300 chars

Use pytest fixtures for common test data.
Mock `call_models_parallel` in pipeline tests — don't make real API calls.

CRITICAL: Use `unittest.mock.AsyncMock` for mocking async functions.
Import path for mocking: "backend.trading_council.call_models_parallel"

VERIFY:
- `uv run pytest backend/tests/test_trading.py -v` — all tests pass
- No real API calls made during tests
```

### Verification Criteria:
- [ ] 18 tests, all passing
- [ ] Parsing tests cover all 3 fallback tiers
- [ ] Consensus tests cover unanimous, majority, and split scenarios
- [ ] Pipeline tests mock API calls
- [ ] No network calls during test execution

---

## STEP 6: Documentation update

**Goal:** Update CLAUDE.md and README.md with Trading Council documentation.

**Context for Claude Code:** `CLAUDE.md`, `README.md`, all new files

### Prompt:

```
Update documentation for the Trading Council feature.

FILE 1: CLAUDE.md — ADD a new section "## Trading Council Mode" after the existing content:

Document:
- Purpose: programmatic endpoint for Python trading engine
- New files: trading_models.py, trading_config.py, trading_council.py, prompts/trading/
- API: POST /api/trading/analyze
- Call flow: snapshot → build_prompts → call_models_parallel → parse → aggregate → alert
- Trigger types and which skip Stage 2
- Config: TRADING_MODELS weights, temperature, timeout
- JSON parsing fallback strategy
- Cost estimate: ~$2-5/day for 5-15 calls
- Integration: Layer 1 Python engine → Council → Telegram alert

FILE 2: README.md — ADD a section "## Trading Council Mode" before "## Tech Stack":

Brief user-facing docs:
- What it is (programmatic LLM council for trading signals)
- How to enable (TRADING_COUNCIL_ENABLED=true in .env, plus 3 API keys)
- Sample curl command from spec §11
- Expected response format (abbreviated)
- Note: does not affect web UI / chat functionality

CRITICAL: Do NOT modify any existing content in either file. Only ADD new sections.

VERIFY:
- CLAUDE.md contains "Trading Council Mode" section
- README.md contains "Trading Council Mode" section with curl example
- Existing content in both files unchanged
```

### Verification Criteria:
- [ ] Both docs updated with new sections
- [ ] Curl example in README matches spec §11
- [ ] Existing documentation unchanged
- [ ] New developer can understand how to enable and test Trading Council

---

## Risk Areas

| Risk | Mitigation |
|------|-----------|
| `llm_providers.py` doesn't exist yet | Step 0 handles both cases (exists vs. create) |
| Anthropic system prompt format | Explicit CRITICAL warning in Step 0 — system goes in top-level field |
| Google Gemini API format differences | Step 0 documents contents/role mapping |
| Models return non-JSON responses | 3-tier fallback parser + HOLD default (safety net) |
| Partial MarketSnapshot payloads | All nested fields Optional in Pydantic models |
| Stage 2 adds latency for time-sensitive triggers | SKIP_STAGE2_TRIGGERS set per spec |
| Test isolation from real APIs | AsyncMock throughout test suite |

---

## Post-Implementation Checklist

After all 6 steps complete:

```bash
# 1. Server starts clean
uv run python -m backend.main

# 2. Health check
curl http://localhost:8001/

# 3. Existing API works
curl http://localhost:8001/api/conversations

# 4. Trading endpoint works (will return 500 without API keys — OK)
curl -X POST http://localhost:8001/api/trading/analyze \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2026-03-11T16:45:00+02:00","trigger_type":"OVERRIDE_STATE_CHANGE","trigger_details":"Override ON → OFF_WARNING","override":{"state":"OFF_WARNING","previous_state":"ON","z15":-0.45,"z30":-0.82,"vix":25.15,"rebound_pct":8.2},"ema_gate":{"state_4h":"BEAR","state_m5":"PULLBACK","trend_score":0.32},"geostress":{"active":false,"score":1.2},"cross_asset":{"btc":{"price":69500,"change_1h_pct":-0.8},"coin_leading":true},"session":{"zone":"POWER_HOUR","zone_reliability":"HIGH","minutes_to_close":55}}'

# 5. All tests pass
uv run pytest backend/tests/test_trading.py -v

# 6. With real API keys — full E2E test
# Expected: 3 model responses + consensus + alert_text, latency < 10s
```
