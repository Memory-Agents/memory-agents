# Memory Augmented Agentic LLM: Initial Setup

This is the initial setup for the ADL project group 14: Memory Augmented Agentic LLM.

It consists of service dependencies provided by a Docker Compose configuration and the memory agents backend.

The memory agents backend includes the LongMemEval benchmark: https://github.com/xiaowu0162/LongMemEval

## Setup

### Requirements

- Docker
- Python (for executing script)
- `uv`: https://docs.astral.sh/uv/

### Step 1: Install `uv`

Install `uv` according to your OS. See [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Install dependencies

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

## Folder structure

```
.
├── .github
│   └── workflows
│       ├── evaluate_baseline.yml       # Github workflow to evaluate the baseline agent
│       ├── evaluate_graphiti_vdb.yml   # Github workflow to evaluate the graphiti agent with a vector database
│       └── evaluate_graphiti.yml       # Github workflow to evaluate the graphiti agent
├── .vscode
│   └── settings.json                   # VSCode settings for the project
├── memory_agents                       # The main application directory
│   ├── core
│   │   ├── agents
│   │   │   ├── baseline.py             # A simple agent with in-memory message history
|   |   |   ├── baseline_vdb.py         # A simple agent  that uses a vector database for memory
│   │   │   ├── graphiti.py             # An agent that uses graphiti for memory
│   │   │   └── graphiti_vdb.py         # An agent that uses graphiti and a vector database for memory
│   │   ├── config.py                   # Configuration for the agents
│   │   └── run_agent.py                # Helper function to run the agents
│   ├── longmemeval                     # Benchmark for evaluating long-term memory in agents
│   │   ├── src
│   │   │   ├── evaluation              # Scripts for evaluating QA and retrieval metrics
│   │   │   ├── generation              # Scripts for running answer generation
│   │   │   ├── index_expansion         # Scripts for expanding the index with different methods
│   │   │   ├── retrieval               # Scripts for running retrieval
│   │   │   └── utils                   # Utility scripts
│   │   ├── data                        # Data for the benchmark
│   │   ├── answerGeneration.py         # Script for generating answers
│   │   └── README.md
│   ├── tests
│   │   ├── agent_initialization_test.py # Tests for agent initialization
│   │   └── agent_query_test.py         # Tests for querying the agents
│   ├── main.py                         # Entry point for running the agents
│   ├── pyproject.toml                  # Project metadata and dependencies
│   └── README.md                       # README for the memory_agents application
├── shared                              # Directory for shared files (currently empty)
├── .env.example                        # Example environment file
├── docker-compose.yml                  # Docker compose file for running services
├── LICENSE                             # Project license
└── README.md                           # Main README for the project
```

## Python formatting and code quality checks

```
cd memory_agents
ruff format .
```

```
cd memory_agents
ruff check .
ruff check --fix
```

## Run PyTest tests

```
cd memory_agents
uv run pytest .
```
