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

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, direct API calls to OpenAI/Anthropic/Google
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript
