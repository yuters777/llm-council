# CLAUDE.md - Technical Notes for LLM Council

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites.

## Architecture

### Backend Structure (`backend/`)

**`config.py`**
- Contains `COUNCIL_MODELS` (list of `ModelConfig` objects)
- Contains `CHAIRMAN_MODEL` (model that synthesizes final answer)
- Contains `TITLE_MODEL` (fast model for generating conversation titles)
- Uses environment variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` from `.env`
- Backend runs on **port 8001** (NOT 8000 - user had another app on 8000)
- Validates API keys on startup for configured providers

**`llm_providers.py`** - Provider Abstraction Layer
- `ModelConfig`: Dataclass with `provider`, `model`, `temperature`, `max_tokens`
- `Attachment`: TypedDict for file attachments (`type`, `media_type`, `data`, `filename`)
- `call_model()`: Single async model query that routes to the correct provider
- `call_models_parallel()`: Parallel queries using `asyncio.gather()`
- `validate_api_keys()`: Validates required API keys for configured models
- File attachment support:
  - `_format_openai_content()`: Formats images as data URLs, PDFs as image_url
  - `_format_anthropic_content()`: Native image and PDF document support
  - `_format_google_parts()`: Inline data format for images and PDFs
- Provider implementations:
  - `_call_openai()`: OpenAI Chat Completions API (with vision support)
  - `_call_anthropic()`: Anthropic Messages API (with image/PDF support)
  - `_call_google()`: Google Generative Language API (Gemini with multimodal)
- Supported file types:
  - Images: jpg, jpeg, png, gif, webp
  - Documents: pdf, txt, csv, json, md
- Max file size: 20MB per file
- Returns string response or None on failure; graceful degradation

**`council.py`** - The Core Logic
- `stage1_collect_responses(user_query, attachments)`: Parallel queries to all council models (with optional attachments)
- `stage2_collect_rankings()`:
  - Anonymizes responses as "Response A, B, C, etc."
  - Creates `label_to_model` mapping for de-anonymization
  - Prompts models to evaluate and rank (with strict format requirements)
  - Returns tuple: (rankings_list, label_to_model_dict)
  - Each ranking includes both raw text and `parsed_ranking` list
- `stage3_synthesize_final()`: Chairman synthesizes from all responses + rankings
- `parse_ranking_from_text()`: Extracts "FINAL RANKING:" section, handles both numbered lists and plain format
- `calculate_aggregate_rankings()`: Computes average rank position across all peer evaluations
- `generate_conversation_title()`: Uses TITLE_MODEL for fast title generation

**`storage.py`**
- JSON-based conversation storage in `data/conversations/`
- Each conversation: `{id, created_at, messages[]}`
- Assistant messages contain: `{role, stage1, stage2, stage3}`
- Note: metadata (label_to_model, aggregate_rankings) is NOT persisted to storage, only returned via API

**`main.py`**
- FastAPI app with CORS enabled for localhost:5173 and localhost:3000
- POST `/api/conversations/{id}/message` returns metadata in addition to stages
- Accepts optional `attachments` array in request body (base64-encoded files)
- `AttachmentModel`: Pydantic model with validation for media_type and file size
- Metadata includes: label_to_model mapping and aggregate_rankings

### Frontend Structure (`frontend/src/`)

**`App.jsx`**
- Main orchestration: manages conversations list and current conversation
- Handles message sending and metadata storage
- Important: metadata is stored in the UI state for display but not persisted to backend JSON

**`components/ChatInterface.jsx`**
- Multiline textarea (3 rows, resizable)
- Enter to send, Shift+Enter for new line
- User messages wrapped in markdown-content class for padding
- File upload support:
  - Attach button (paperclip icon) for file selection
  - Preview area for attached files (thumbnails for images, icons for documents)
  - Remove button on each attachment
  - File validation (type, size) with error messages

**`api.js`**
- `sendMessage(conversationId, content, files)`: Sends message with optional file attachments
- `sendMessageStream(conversationId, content, onEvent, files)`: Streaming version
- `validateFile(file)`: Client-side validation for file type and size
- `fileToBase64(file)`: Converts File to base64 string
- Supported types exported for UI validation

**`components/Stage1.jsx`**
- Tab view of individual model responses
- ReactMarkdown rendering with markdown-content wrapper

**`components/Stage2.jsx`**
- **Critical Feature**: Tab view showing RAW evaluation text from each model
- De-anonymization happens CLIENT-SIDE for display (models receive anonymous labels)
- Shows "Extracted Ranking" below each evaluation so users can validate parsing
- Aggregate rankings shown with average position and vote count
- Explanatory text clarifies that boldface model names are for readability only

**`components/Stage3.jsx`**
- Final synthesized answer from chairman
- Green-tinted background (#f0fff0) to highlight conclusion

**Styling (`*.css`)**
- Light mode theme (not dark mode)
- Primary color: #4a90e2 (blue)
- Global markdown styling in `index.css` with `.markdown-content` class
- 12px padding on all markdown content to prevent cluttered appearance

## Key Design Decisions

### Provider Abstraction
The `llm_providers.py` module provides a clean abstraction over different LLM providers:
- Each provider has its own implementation handling API-specific details
- Models are configured using `ModelConfig(provider="openai", model="gpt-4.1")`
- The rest of the codebase doesn't need to know provider-specific details
- Easy to add new providers (just add a new `_call_*` function)

### Stage 2 Prompt Format
The Stage 2 prompt is very specific to ensure parseable output:
```
1. Evaluate each response individually first
2. Provide "FINAL RANKING:" header
3. Numbered list format: "1. Response C", "2. Response A", etc.
4. No additional text after ranking section
```

This strict format allows reliable parsing while still getting thoughtful evaluations.

### De-anonymization Strategy
- Models receive: "Response A", "Response B", etc.
- Backend creates mapping: `{"Response A": "openai/gpt-4.1", ...}`
- Frontend displays model names in **bold** for readability
- Users see explanation that original evaluation used anonymous labels
- This prevents bias while maintaining transparency

### File Attachment Architecture
Attachments flow through the system as follows:
1. Frontend converts files to base64 and sends with message
2. Backend validates file type and size via Pydantic validators
3. Stage 1 passes attachments to all council models
4. Each provider formats attachments according to its API:
   - **OpenAI**: `image_url` with data URLs, PDFs also as image_url
   - **Anthropic**: Native `image` and `document` content blocks
   - **Google**: `inlineData` parts with mimeType
5. Stages 2 and 3 do NOT receive attachments (evaluate text responses only)
6. Text-based documents (txt, csv, json, md) are decoded and included inline

### Error Handling Philosophy
- Continue with successful responses if some models fail (graceful degradation)
- Never fail the entire request due to single model failure
- Log errors but don't expose to user unless all models fail
- API keys are validated on startup with clear error messages
- Invalid attachments are skipped with a warning, not rejected

### UI/UX Transparency
- All raw outputs are inspectable via tabs
- Parsed rankings shown below raw text for validation
- Users can verify system's interpretation of model outputs
- This builds trust and allows debugging of edge cases

## Important Implementation Details

### Relative Imports
All backend modules use relative imports (e.g., `from .config import ...`) not absolute imports. This is critical for Python's module system to work correctly when running as `python -m backend.main`.

### Port Configuration
- Backend: 8001 (changed from 8000 to avoid conflict)
- Frontend: 5173 (Vite default)
- Update both `backend/main.py` and `frontend/src/api.js` if changing

### Markdown Rendering
All ReactMarkdown components must be wrapped in `<div className="markdown-content">` for proper spacing. This class is defined globally in `index.css`.

### Model Configuration
Models are configured in `backend/config.py` using `ModelConfig` objects. Each model specifies:
- `provider`: One of "openai", "anthropic", "google"
- `model`: Provider-specific model name (e.g., "gpt-4.1", "claude-sonnet-4-20250514", "gemini-2.0-flash")
- `temperature`: Sampling temperature (default 1.0)
- `max_tokens`: Maximum response tokens (default 4096)

Chairman can be same or different from council members. The current default is Gemini as chairman.

### API Message Format Conversion
Each provider has different message format requirements:
- **OpenAI**: Standard `[{"role": "user", "content": "..."}]` format
- **Anthropic**: System message must be a separate `system` field, not in messages array
- **Google**: Uses `contents` with `parts` structure, `assistant` role becomes `model`

The provider implementations handle these conversions transparently.

## Common Gotchas

1. **Module Import Errors**: Always run backend as `python -m backend.main` from project root, not from backend directory
2. **CORS Issues**: Frontend must match allowed origins in `main.py` CORS middleware
3. **Ranking Parse Failures**: If models don't follow format, fallback regex extracts any "Response X" patterns in order
4. **Missing Metadata**: Metadata is ephemeral (not persisted), only available in API responses
5. **Missing API Keys**: If a provider's API key is not set, the startup will print a warning. Models from that provider will fail.

## Future Enhancement Ideas

- Configurable council/chairman via UI instead of config file
- Streaming responses instead of batch loading
- Export conversations to markdown/PDF
- Model performance analytics over time
- Custom ranking criteria (not just accuracy/insight)
- Support for reasoning models (o1, etc.) with special handling
- Add more providers (local models, other cloud providers)

## Data Flow Summary

```
User Query + Optional Attachments
    ↓
Stage 1: Parallel queries via call_models_parallel() WITH attachments → [individual responses]
    ↓
Stage 2: Anonymize → Parallel ranking queries (text only) → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis (text only) via call_model()
    ↓
Return: {stage1, stage2, stage3, metadata}
    ↓
Frontend: Display with tabs + validation UI
```

The entire flow is async/parallel where possible to minimize latency.

## Attachment Format (API Request)

```json
{
  "content": "What's in this image?",
  "attachments": [
    {
      "type": "image",
      "media_type": "image/png",
      "data": "<base64-encoded-content>",
      "filename": "screenshot.png"
    },
    {
      "type": "document",
      "media_type": "application/pdf",
      "data": "<base64-encoded-content>",
      "filename": "report.pdf"
    }
  ]
}
```

## Trading Council Mode

### Purpose

A programmatic endpoint (`POST /api/trading/analyze`) for a Python trading engine to get consensus-based trading decisions from multiple LLMs. This is a machine-to-machine API — it does not affect the web UI or chat functionality.

### New Files

- **`backend/trading_models.py`** — Pydantic v2 models for `MarketSnapshot` (input) and `TradingDecision` (output), plus all nested schemas (OverrideData, GeoStressData, EmaGateData, CryptoOverrideData, CrossAssetData, SessionData, ModelVote, ConsensusResult, TradingMeta)
- **`backend/trading_config.py`** — Trading-specific configuration: `TRADING_MODELS` (3 weighted council members), `SKIP_STAGE2_TRIGGERS`, timeouts, fallback defaults
- **`backend/trading_council.py`** — Core logic: prompt building, 3-tier response parsing, consensus aggregation, Telegram alert formatting, and the `analyze_trading()` orchestrator
- **`backend/prompts/trading/`** — 6 Markdown prompt templates (system_base, override_change, geostress_alert, morning_briefing, conflicting_signals, unusual_pattern)
- **`backend/tests/test_trading.py`** — 18 pytest tests covering parsing, consensus, pipeline, models, and alerts

### API Endpoint

```
POST /api/trading/analyze
Content-Type: application/json
Body: MarketSnapshot JSON

Returns: TradingDecision JSON
```

### Call Flow

```
MarketSnapshot (from Python engine)
    ↓
build_prompts() → system prompt + user message with snapshot data
    ↓
call_models_parallel() → 3 models (Claude, GPT, Gemini) respond in parallel
    ↓
parse_trading_response() → 3-tier fallback: direct JSON → markdown block → regex → HOLD fallback
    ↓
Stage 2 peer adjustment (skipped for time-sensitive triggers)
    ↓
aggregate_decisions() → weighted consensus (claude=0.40, gpt=0.35, gemini=0.25)
    ↓
format_telegram_alert() → < 300 char alert with emoji prefix
    ↓
TradingDecision (returned to caller)
```

### Trigger Types

| Trigger | Skips Stage 2 | Description |
|---------|--------------|-------------|
| `OVERRIDE_STATE_CHANGE` | Yes | Override regime transition |
| `GEOSTRESS_ALERT` | Yes | Cross-asset stress detected |
| `UNUSUAL_PATTERN` | Yes | Anomalous market pattern |
| `CONFLICTING_SIGNALS` | No | Contradictory framework signals |
| `MORNING_BRIEFING` | No | Daily pre-market analysis |
| `EARNINGS_PROXIMITY` | No | Earnings event approaching |

### Configuration

- **Models**: 3 council members in `TRADING_MODELS` dict, each with a weight and `ModelConfig`
- **Temperature**: 0.3 (deterministic analysis, not creative writing)
- **Timeout**: 30 seconds per call
- **Fallback**: On parse failure → HOLD with 0.3 confidence (safety net)
- **Feature toggle**: `TRADING_COUNCIL_ENABLED` env var (default: true)

### JSON Parsing Fallback Strategy

1. **Tier 1**: Direct `json.loads()` on raw response
2. **Tier 2**: Extract JSON from `` ```json ... ``` `` markdown code block
3. **Tier 3**: Regex extraction of `"decision"`, `"confidence"`, etc.
4. **Fallback**: `ModelVote(decision="HOLD", confidence=0.3)` — never act on unparseable output

### Cost Estimate

~$2-5/day for 5-15 calls. Approximate per-call rates (input/output per MTok):
- Claude: $3 / $15
- GPT: $2 / $8
- Gemini: $1.25 / $10

### Integration

```
Layer 1 Python Engine → POST /api/trading/analyze → Trading Council → TradingDecision
                                                                    → alert_text → Telegram
```
