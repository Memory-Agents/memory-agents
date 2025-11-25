# Memory Augmented Agentic LLM: Initial Setup

This is the initial setup for the ADL project group 14: Memory Augmented Agentic LLM.

It consists of service dependencies provided by a Docker Compose configuration and the memory agents backend.

The memory agents backend includes the LongMemEval benchmark: https://github.com/xiaowu0162/LongMemEval

## Setup

#### Requirements
- Docker
- Python (for executing script)
- `uv`: https://docs.astral.sh/uv/

#### Step 1: Install `uv`

Install `uv` according to your OS. See [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


#### Step 2: Install dependencies

```bash
cd memory_agents
uv sync --all-groups
```

### Step 3: Set environment variables

#### Using a `.env` file (recommended)

1. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

2. Open `.env` and insert your actual API key:

```bash
# .env
OPENAI_API_KEY=your-actual-openai-api-key-here
```

#### Set environment variables manually (alternative)

```bash
# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY="your-api-key-here"
```

#### Service ports

- Openwebui (8080)
- ...

## How to run

```bash
docker compose up
TODO
```