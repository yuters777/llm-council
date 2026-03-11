# ARCHITECTURE.md — LLM Council Backend

## 1. System Overview

LLM Council runs two independent modes on a single FastAPI server (port 8001):

- **General Council** — A 3-stage deliberation system for the web UI. Multiple LLMs answer a user question (Stage 1), anonymously peer-review each other (Stage 2), then a chairman synthesizes the final answer (Stage 3). Conversations are persisted to JSON files.
- **Trading Council** — A machine-to-machine endpoint (`POST /api/trading/analyze`) for a Python trading engine. Three weighted LLMs analyze a `MarketSnapshot` in parallel, votes are parsed and aggregated into a consensus `TradingDecision`, and a Telegram alert string is generated. No conversation persistence.

**Shared infrastructure:** Both modes use the same provider abstraction layer (`llm_providers.py`) for all LLM calls, sharing a single `httpx.AsyncClient` with connection pooling and retry logic. Both use `ModelConfig` to define models and `call_models_parallel()` for concurrent queries.

---

## 2. File Map

| File | Purpose |
|------|---------|
| `backend/__init__.py` | Package marker (single docstring) |
| `backend/llm_providers.py` | Provider-agnostic LLM interface: HTTP client, retry logic, OpenAI/Anthropic/Google implementations |
| `backend/config.py` | General Council configuration: council models, chairman, title model, data directory |
| `backend/council.py` | 3-stage General Council orchestration: collect → rank → synthesize |
| `backend/storage.py` | JSON-based conversation persistence in `data/conversations/` |
| `backend/main.py` | FastAPI app with all HTTP endpoints and Pydantic request/response models |
| `backend/trading_models.py` | Pydantic v2 models for `MarketSnapshot` (input) and `TradingDecision` (output) |
| `backend/trading_config.py` | Trading Council configuration: models with weights, timeouts, feature toggle |
| `backend/trading_council.py` | Trading Council pipeline: prompts → call → parse → aggregate → alert |
| `backend/tests/__init__.py` | Test package marker |
| `backend/tests/test_trading.py` | 18 pytest tests for the trading council |

---

## 3. Provider Layer (`llm_providers.py`)

### ModelConfig

```python
@dataclass
class ModelConfig:
    provider: Literal["openai", "anthropic", "google"]
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096
```

Hashed by `(provider, model)`. Has a `display_name` property returning `"{provider}/{model}"`.

### Attachment TypedDict

```python
class Attachment(TypedDict):
    type: Literal["image", "document"]
    media_type: str
    data: str        # base64-encoded
    filename: str
```

### Core Functions

```python
async def call_model(
    model_config: ModelConfig,
    messages: List[Dict[str, Any]],
    timeout: float = 120.0,
    attachments: Optional[List[Attachment]] = None,
) -> Optional[str]
```

```python
async def call_models_parallel(
    model_configs: List[ModelConfig],
    messages: List[Dict[str, Any]],
    timeout: float = 120.0,
    attachments: Optional[List[Attachment]] = None,
) -> Dict[ModelConfig, Optional[str]]
```

```python
def validate_api_keys(configs: List[ModelConfig]) -> None
```

### Connection Pooling

A module-level `httpx.AsyncClient` is lazily created via `get_http_client()`:

- Timeout: 120s overall, 10s connect
- Limits: 100 max connections, 20 max keepalive
- Closed on app shutdown via `close_http_client()`

### Retry Logic (`_request_with_retry`)

- Max retries: `DEFAULT_MAX_RETRIES = 3`
- Base delay: `DEFAULT_BASE_DELAY = 1.0` seconds
- Backoff: exponential (`base_delay * 2^attempt`)
- Retryable status codes: `{429, 502, 503}`
- Also retries on `httpx.TimeoutException` and `httpx.RequestError`

### Provider-Specific Quirks

**Anthropic** — System messages must be extracted from the messages array and passed as a top-level `system` field in the payload. Attachments placed *before* text content. Uses `x-api-key` header and `anthropic-version: 2023-06-01`.

**Google (Gemini)** — Role `assistant` maps to `model`. System instructions go in a top-level `systemInstruction.parts` field. Uses URL template with model name: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`. API key is a query parameter, not a header.

**OpenAI** — Standard Chat Completions format. PDFs sent as both a text note and an `image_url` data URL. Uses `Authorization: Bearer` header.

### File Constants

- `MAX_FILE_SIZE_BYTES`: 20 MB (`20 * 1024 * 1024`)
- `SUPPORTED_IMAGE_TYPES`: `{image/jpeg, image/png, image/gif, image/webp}`
- `SUPPORTED_DOCUMENT_TYPES`: `{application/pdf, text/plain, text/csv, application/json, text/markdown}`

---

## 4. General Council Flow (`council.py`)

### Stage 1 — Collect Responses

```python
async def stage1_collect_responses(
    user_query: str,
    attachments: Optional[List[Attachment]] = None,
) -> List[Dict[str, Any]]
```

- Sends `user_query` (with optional attachments) to all `COUNCIL_MODELS` in parallel
- Returns: `[{"model": "openai/gpt-4.1", "response": "..."}, ...]`
- Failed models are silently excluded

### Stage 2 — Anonymized Peer Ranking

```python
async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]
```

- Anonymizes Stage 1 responses as "Response A", "Response B", etc.
- Creates `label_to_model` mapping: `{"Response A": "openai/gpt-4.1", ...}`
- Prompts require `FINAL RANKING:` header followed by numbered list
- No attachments passed — text evaluation only
- Returns: `([{"model": ..., "ranking": ..., "parsed_ranking": [...]}, ...], label_to_model)`

### Ranking Parser

```python
def parse_ranking_from_text(ranking_text: str) -> List[str]
```

- Splits on `"FINAL RANKING:"`, extracts numbered `"Response X"` patterns via regex
- Fallback: finds all `"Response [A-Z]"` patterns anywhere in text

### Aggregate Rankings

```python
def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str],
) -> List[Dict[str, Any]]
```

- Returns: `[{"model": ..., "average_rank": 1.67, "rankings_count": 3}, ...]` sorted by `average_rank` ascending

### Stage 3 — Chairman Synthesis

```python
async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
) -> Dict[str, Any]
```

- Sends all Stage 1 responses + Stage 2 rankings to `CHAIRMAN_MODEL`
- Returns: `{"model": "google/gemini-2.0-flash", "response": "..."}`
- Fallback on failure: returns error message string

### Full Pipeline

```python
async def run_full_council(
    user_query: str,
    attachments: Optional[List[Attachment]] = None,
) -> Tuple[List, List, Dict, Dict]
```

Returns `(stage1_results, stage2_results, stage3_result, metadata)` where metadata contains `label_to_model` and `aggregate_rankings`.

### Title Generation

```python
async def generate_conversation_title(user_query: str) -> str
```

Uses `TITLE_MODEL` with 30s timeout. Returns 3-5 word title or `"New Conversation"` on failure.

---

## 5. Trading Council Flow (`trading_council.py`)

### Full Pipeline

```
MarketSnapshot
  → build_prompts()      → (system_prompt, messages)
  → call_models_parallel() → Dict[ModelConfig, Optional[str]]
  → parse_trading_response() per model → Dict[str, ModelVote]
  → _stage2_peer_adjustment() (if not time-sensitive trigger)
  → aggregate_decisions() → ConsensusResult
  → format_telegram_alert() → str (< 300 chars)
  → TradingDecision
```

### Prompt Building

```python
def build_prompts(snapshot: MarketSnapshot) -> tuple[str, list[dict]]
```

- Loads `system_base.md` as system message
- Loads trigger-specific template, replaces `{snapshot_json}` with `snapshot.model_dump_json(indent=2)`
- `MORNING_BRIEFING` also replaces `{news_json}` and `{movers_json}` (currently with `"{}"`)
- Returns `(system_prompt, [{"role": "system", ...}, {"role": "user", ...}])`

### Response Parsing — 3-Tier Fallback

```python
def parse_trading_response(raw_text: str, model_name: str) -> ModelVote
```

| Tier | Method | Trigger |
|------|--------|---------|
| 1 | `json.loads(raw_text.strip())` | Direct JSON response |
| 2 | Regex ```` ```json ... ``` ```` → `json.loads()` | Markdown-wrapped JSON |
| 3 | Individual regex: `"decision": "X"`, `"confidence": N`, `"reasoning": "..."`, `"key_factor": "..."`, `"risk_flag": "..."` | Partial JSON fragments |
| Fallback | `ModelVote(decision="HOLD", confidence=0.3)` | All tiers failed |

Confidence values are clamped to `[0.0, 1.0]` via `_clamp_confidence()`.

### Consensus Aggregation

```python
def aggregate_decisions(votes: dict[str, ModelVote]) -> ConsensusResult
```

**Decision selection:**
- Majority vote wins
- 3-way split (each model different) → `"HOLD"`

**Weighted confidence:**
- Each model's confidence is multiplied by its weight from `TRADING_MODELS`
- Weights: claude=0.40, gpt=0.35, gemini=0.25
- Formula: `sum(vote.confidence * weight for each model)` (the code divides by total_weight then multiplies back, effectively yielding the weighted sum)

**Consensus strength:** `"UNANIMOUS"` if all agree, else `"{count}/{total}"` (e.g., `"2/3"`)

**Dissent summary:** Lists dissenting models with their decisions and key factors.

### Stage 2 Peer Adjustment

```python
async def _stage2_peer_adjustment(
    votes: dict[str, ModelVote],
    original_messages: list[dict],
) -> dict[str, ModelVote]
```

- Skipped for triggers in `SKIP_STAGE2_TRIGGERS`
- Models agreeing with majority: confidence × 1.05 (capped at 1.0)
- Dissenters: confidence × 0.95 (floored at 0.0)

### Telegram Alert Formatting

```python
def format_telegram_alert(
    snapshot: MarketSnapshot,
    result: ConsensusResult,
    votes: dict[str, ModelVote],
) -> str
```

**Emoji map:**

| Trigger | Emoji |
|---------|-------|
| `OVERRIDE_STATE_CHANGE` (state ON/ON_CONFIRMED) | 🟢 |
| `OVERRIDE_STATE_CHANGE` (other) | 🔴 |
| `GEOSTRESS_ALERT` | 🚨 |
| `CONFLICTING_SIGNALS` | ⚠️ |
| `MORNING_BRIEFING` | 📋 |
| `UNUSUAL_PATTERN` | 🔍 |
| `EARNINGS_PROXIMITY` | 📅 |
| Unknown | 💬 |

Format: `"{emoji} {trigger}\n{decision} ({strength}, conf {pct}){vix}\nDissent: {summary}"`

Hard cap: 300 characters. Dissent truncated to 100 characters if needed.

### Cost Estimation

```python
def _estimate_cost(input_tokens: int, output_tokens: int) -> float
```

Per-million-token rates (split equally across 3 models):
- Claude: $3 input / $15 output
- GPT: $2 input / $8 output
- Gemini: $1.25 input / $10 output

Token estimate: ~4 characters per token.

---

## 6. Data Models (`trading_models.py`)

### Input — MarketSnapshot Tree

```
MarketSnapshot
├── timestamp: str
├── trigger_type: TRIGGER_TYPES (Literal)
├── trigger_details: Optional[str]
├── override: Optional[OverrideData]
│   ├── state, previous_state: Optional[str]
│   ├── z15, z30, override_score, vix, vix_5min_ago, vvix: Optional[float]
│   ├── rebound_pct, term_structure, vix_fatigue_modifier: Optional[float]
├── geostress: Optional[GeoStressData]
│   ├── active: Optional[bool]
│   ├── score: Optional[float]
│   ├── components: Optional[GeoStressComponents]
│       ├── z_dvix_5, z_dvvix_5, z_gold_5, z_oil_5_abs: Optional[float]
│       ├── z_jpy_dxy, breadth_shock: Optional[float]
├── ema_gate: Optional[EmaGateData]
│   ├── state_4h, state_m5, m5_substate: Optional[str]
│   ├── trend_score, adx: Optional[float]
│   ├── ticker_states: Optional[dict[str, TickerEmaState]]
│       ├── state_4h (alias "4h"), state_m5 (alias "m5"): Optional[str]
│       ├── score: Optional[float]
├── crypto_override: Optional[CryptoOverrideData]
│   ├── state: Optional[str]
│   ├── dvol, ethdvol, btc_funding_rate, hierarchical_weight: Optional[float]
├── cross_asset: Optional[CrossAssetData]
│   ├── btc, eth: Optional[AssetPrice]
│   │   ├── price, change_1h_pct, volume_vs_median: Optional[float]
│   ├── oil_wti, oil_change_30m_sigma, gold: Optional[float]
│   ├── coin_leading: Optional[bool]
│   ├── ibit_vs_btc_divergence, china_adrs: Optional[float]
├── session: Optional[SessionData]
│   ├── zone, zone_reliability, day_of_week: Optional[str]
│   ├── minutes_to_close: Optional[int]
│   ├── is_event_day, event_quarantine: Optional[bool]
├── recent_alerts: Optional[list[RecentAlert]]
    ├── time, type, details, ticker, timeframe: Optional[str]
```

**TRIGGER_TYPES:** `"OVERRIDE_STATE_CHANGE" | "GEOSTRESS_ALERT" | "CONFLICTING_SIGNALS" | "MORNING_BRIEFING" | "UNUSUAL_PATTERN" | "EARNINGS_PROXIMITY"`

### Output — TradingDecision Tree

```
TradingDecision
├── timestamp: str = ""
├── trigger_type: str = ""
├── decision: str = "HOLD"
├── confidence: float = 0.5
├── reasoning: str = ""
├── council_votes: dict[str, ModelVote] = {}
│   └── ModelVote
│       ├── decision: DECISION_TYPES = "HOLD"
│       ├── confidence: float (0.0–1.0, default 0.5)
│       ├── key_factor: str = ""
│       ├── risk_flag: Optional[str] = None
│       ├── reasoning: str = ""
├── consensus: str = ""
├── consensus_strength: str = ""
├── dissent_summary: str = ""
├── action_items: list[str] = []
├── alert_text: str = ""
├── meta: TradingMeta
    ├── total_tokens: int = 0
    ├── cost_usd: float = 0.0
    ├── latency_ms: int = 0
    ├── models_used: list[str] = []
```

**DECISION_TYPES:** `"BUY" | "SELL" | "HOLD" | "EXIT" | "REDUCE" | "CAUTIOUS_SHORT" | "CAUTIOUS_LONG"`

**ConsensusResult** (internal, not returned directly):

```
ConsensusResult
├── decision: str = "HOLD"
├── confidence: float = 0.5
├── consensus: str = ""           # e.g. "2/3_HOLD"
├── consensus_strength: str = ""  # e.g. "2/3" or "UNANIMOUS"
├── dissent_summary: str = ""
```

All models use `ConfigDict(populate_by_name=True)`.

---

## 7. Configuration

### `config.py` — General Council

| Setting | Value |
|---------|-------|
| `COUNCIL_MODELS` | `[openai/gpt-4.1, anthropic/claude-sonnet-4-20250514, google/gemini-2.0-flash]` (temp=1.0, max_tokens=4096) |
| `CHAIRMAN_MODEL` | `google/gemini-2.0-flash` (temp=1.0, max_tokens=4096) |
| `TITLE_MODEL` | `google/gemini-2.0-flash` (temp=1.0, max_tokens=100) |
| `DATA_DIR` | `"data/conversations"` |

Validates API keys on import via `_validate_on_startup()`. Catches `ValueError` and prints warning.

### `trading_config.py` — Trading Council

| Setting | Value |
|---------|-------|
| `TRADING_ENABLED` | `os.getenv("TRADING_COUNCIL_ENABLED", "true")` — accepts `true/1/yes` |
| `TRADING_TIMEOUT_SECONDS` | `30.0` |
| `FALLBACK_DECISION` | `"HOLD"` |
| `FALLBACK_CONFIDENCE` | `0.3` |
| `PROMPTS_DIR` | `"backend/prompts/trading"` |
| `SKIP_STAGE2_TRIGGERS` | `{"OVERRIDE_STATE_CHANGE", "GEOSTRESS_ALERT", "UNUSUAL_PATTERN"}` |

**Trading Models:**

| Key | Provider | Model | Weight | Temperature | Max Tokens |
|-----|----------|-------|--------|-------------|------------|
| `claude` | anthropic | `claude-sonnet-4-20250514` | 0.40 | 0.3 | 1000 |
| `gpt` | openai | `gpt-4.1` | 0.35 | 0.3 | 1000 |
| `gemini` | google | `gemini-3-pro-preview` | 0.25 | 0.3 | 1000 |

### Environment Variables

| Variable | Used By | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | `llm_providers.py` | If any OpenAI model is configured |
| `ANTHROPIC_API_KEY` | `llm_providers.py` | If any Anthropic model is configured |
| `GOOGLE_API_KEY` | `llm_providers.py` | If any Google model is configured |
| `TRADING_COUNCIL_ENABLED` | `trading_config.py` | No (default: `true`) |

---

## 8. API Surface (`main.py`)

| Method | Path | Request Body | Response | Description |
|--------|------|-------------|----------|-------------|
| `GET` | `/` | — | `{"status": "ok", "service": "LLM Council API", "trading_enabled": bool}` | Health check |
| `GET` | `/api/conversations` | — | `List[ConversationMetadata]` | List conversations (id, created_at, title, message_count) |
| `POST` | `/api/conversations` | `CreateConversationRequest` (empty) | `Conversation` | Create new conversation |
| `GET` | `/api/conversations/{conversation_id}` | — | `Conversation` | Get conversation with all messages |
| `POST` | `/api/conversations/{conversation_id}/message` | `SendMessageRequest` | `{stage1, stage2, stage3, metadata}` | Send message, run full council |
| `POST` | `/api/conversations/{conversation_id}/message/stream` | `SendMessageRequest` | SSE `StreamingResponse` | Send message, stream stages as SSE events |
| `POST` | `/api/trading/analyze` | `MarketSnapshot` | `TradingDecision` | Trading council analysis |

### Request Models

**`SendMessageRequest`**: `content: str`, `attachments: Optional[List[AttachmentModel]]`

**`AttachmentModel`**: `type: Literal["image", "document"]`, `media_type: str` (validated against supported types), `data: str` (base64, validated against 20MB limit), `filename: str`

### SSE Event Types (streaming endpoint)

`stage1_start` → `stage1_complete` → `stage2_start` → `stage2_complete` → `stage3_start` → `stage3_complete` → `title_complete` (if first message) → `complete`

On error: `{"type": "error", "message": "..."}`.

### CORS

Allowed origins: `http://localhost:5173`, `http://localhost:3000`.

### Lifespan

On shutdown: calls `close_http_client()` to clean up the shared httpx client.

---

## 9. Prompt Templates

All templates live in `backend/prompts/trading/`.

| File | Trigger Type(s) | Placeholders |
|------|-----------------|--------------|
| `system_base.md` | All (system message) | None |
| `override_change.md` | `OVERRIDE_STATE_CHANGE` | `{snapshot_json}` |
| `geostress_alert.md` | `GEOSTRESS_ALERT` | `{snapshot_json}` |
| `morning_briefing.md` | `MORNING_BRIEFING` | `{snapshot_json}`, `{news_json}`, `{movers_json}` |
| `conflicting_signals.md` | `CONFLICTING_SIGNALS` | `{snapshot_json}` |
| `unusual_pattern.md` | `UNUSUAL_PATTERN`, `EARNINGS_PROXIMITY` | `{snapshot_json}` |

**Trigger → File mapping** (in `_TRIGGER_FILE_MAP`):

- `OVERRIDE_STATE_CHANGE` → `override_change`
- `GEOSTRESS_ALERT` → `geostress_alert`
- `CONFLICTING_SIGNALS` → `conflicting_signals`
- `MORNING_BRIEFING` → `morning_briefing`
- `UNUSUAL_PATTERN` → `unusual_pattern`
- `EARNINGS_PROXIMITY` → `unusual_pattern` (reuses the same template)

The system prompt (`system_base.md`) defines the framework hierarchy (Override → GeoStress → EMA Gate → Crypto Override → Cross-Asset → Session), decision rules, and requires a strict JSON response format with fields: `decision`, `confidence`, `key_factor`, `risk_flag`, `reasoning`.

Templates are cached in a module-level `_template_cache` dict after first load.

---

## 10. Dependency Graph

```
main.py
├── storage
│   └── config
│       └── llm_providers
├── council
│   ├── llm_providers
│   └── config
│       └── llm_providers
├── trading_council
│   ├── llm_providers
│   ├── trading_config
│   │   └── llm_providers
│   └── trading_models
├── trading_models  (standalone, only pydantic)
├── trading_config
│   └── llm_providers
└── llm_providers   (standalone, only httpx + stdlib)
```

**External dependencies:** `fastapi`, `pydantic`, `httpx`, `uvicorn`, `python-dotenv`

**No provider SDKs** — all LLM calls use raw HTTP via `httpx`. No `openai`, `anthropic`, or `google-generativeai` packages.
