# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT, Google Gemini, Anthropic Claude), you can group them into your "LLM Council". This repo is a simple, local web app that essentially looks like ChatGPT except it sends your query directly to multiple LLMs (OpenAI, Anthropic, and Google), asks them to review and rank each other's work, and finally a Chairman LLM produces the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them all one by one.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs. Under the hood, the LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. **Stage 3: Final response**. The designated Chairman of the LLM Council takes all of the model's responses and compiles them into a single final answer that is presented to the user.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Keys

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI API key (for GPT models)
OPENAI_API_KEY=sk-...

# Anthropic API key (for Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# Google API key (for Gemini models)
GOOGLE_API_KEY=AIza...
```

You only need keys for the providers you configure in your council. Get your API keys at:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/settings/keys
- Google: https://aistudio.google.com/apikey

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council:

```python
from .llm_providers import ModelConfig

# Council members - models that generate and evaluate responses
COUNCIL_MODELS = [
    ModelConfig(provider="openai", model="gpt-4.1"),
    ModelConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    ModelConfig(provider="google", model="gemini-2.0-flash"),
]

# Chairman model - synthesizes the final response
CHAIRMAN_MODEL = ModelConfig(provider="google", model="gemini-2.0-flash")
```

Available providers: `"openai"`, `"anthropic"`, `"google"`

Example model names:
- OpenAI: `"gpt-4.1"`, `"gpt-4o"`, `"gpt-4o-mini"`
- Anthropic: `"claude-sonnet-4-20250514"`, `"claude-3-5-sonnet-20241022"`, `"claude-3-haiku-20240307"`
- Google: `"gemini-2.0-flash"`, `"gemini-1.5-pro"`, `"gemini-1.5-flash"`

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Trading Council Mode

In addition to the interactive web UI, LLM Council includes a **programmatic trading analysis endpoint** designed for machine-to-machine calls from a Python trading engine. This feature does not affect the web UI or chat functionality.

### How to Enable

1. Set API keys in `.env` (same keys used by the chat council):
   ```bash
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=AIza...
   ```

2. Optionally toggle with `TRADING_COUNCIL_ENABLED=true` (enabled by default).

### Usage

Send a `MarketSnapshot` JSON to `POST /api/trading/analyze`:

```bash
curl -X POST http://localhost:8001/api/trading/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2026-03-11T16:45:00+02:00",
    "trigger_type": "OVERRIDE_STATE_CHANGE",
    "trigger_details": "Override ON -> OFF_WARNING",
    "override": {"state": "OFF_WARNING", "previous_state": "ON", "z15": -0.45, "z30": -0.82, "vix": 25.15, "rebound_pct": 8.2},
    "ema_gate": {"state_4h": "BEAR", "state_m5": "PULLBACK", "trend_score": 0.32},
    "geostress": {"active": false, "score": 1.2},
    "cross_asset": {"btc": {"price": 69500, "change_1h_pct": -0.8}, "coin_leading": true},
    "session": {"zone": "POWER_HOUR", "zone_reliability": "HIGH", "minutes_to_close": 55}
  }'
```

### Response (abbreviated)

```json
{
  "timestamp": "2026-03-11T16:45:00+02:00",
  "trigger_type": "OVERRIDE_STATE_CHANGE",
  "decision": "HOLD",
  "confidence": 0.65,
  "reasoning": "Council UNANIMOUS consensus: HOLD. ...",
  "council_votes": {"claude": {...}, "gpt": {...}, "gemini": {...}},
  "consensus_strength": "UNANIMOUS",
  "alert_text": "🔴 Override State Change\nHOLD (UNANIMOUS, conf 65%) | VIX 25.2",
  "meta": {"total_tokens": 1234, "cost_usd": 0.002, "latency_ms": 3500, "models_used": ["claude", "gpt", "gemini"]}
}
```

Three models (Claude Sonnet, GPT-4.1, Gemini Pro) analyze the snapshot in parallel, vote on a trading decision, and return a weighted consensus with a Telegram-ready alert.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, direct API calls to OpenAI/Anthropic/Google
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
