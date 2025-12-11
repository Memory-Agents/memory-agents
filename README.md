# Memory Augmented Agentic LLM: Initial Setup

This is the repository of the ADL project group 14: Memory Augmented Agentic LLM.

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

To run the full stack, you will need to configure environment variables for the different services. This is done by creating a `.env` file from the provided `.env.example` in three different locations. For each of them, you will need to at least provide your `OPENAI_API_KEY`.

1.  **Root Environment:** For the main service Langfuse, managed by the root Docker Compose file.

    ```bash
    # In the project root directory
    cp .env.example .env
    ```
    Then, edit `.env` and set your `OPENAI_API_KEY`.

2.  **Memory Agents Environment:** For the Python-based memory agents.

    ```bash
    # In the project root directory
    cp memory_agents/.env.example memory_agents/.env
    ```
    Then, edit `memory_agents/.env` and set your `OPENAI_API_KEY`. You will also need to add your Langfuse keys if you are using Langfuse for tracing.

3.  **Graphiti MCP Server Environment:** For the Graphiti memory service.

    ```bash
    # In the project root directory
    cp graphiti/mcp_server/.env.example graphiti/mcp_server/.env
    ```
    Then, edit `graphiti/mcp_server/.env` and set your `OPENAI_API_KEY`.

After copying, ensure each `.env` file contains your actual OpenAI API key, for example:
```
# .env (example content)
OPENAI_API_KEY=your-actual-openai-api-key-here
```

#### Set environment variables manually (alternative)

As an alternative to using `.env` files, you can export the variables directly in your shell. You will need to do this for each terminal session where you run a part of the application. However, running the backend services still requires to configure the `docker-compose.yml` in the root directory to read the locally set environment variable.

```bash
# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY="your-api-key-here"
```

### Step 4: Configure Langfuse (Optional)

If you want to use Langfuse for tracing, you need to configure it after starting the services for the first time.

1.  **Open Langfuse:** Once the services are running (see "How to run"), open your web browser and navigate to `http://localhost:3002`.
2.  **Sign Up:** Create a new account. The first user to sign up will become the administrator.
3.  **Create an Organization and Project:**
    *   After logging in, you'll be prompted to create a new **organization**.
    *   Once the organization is created, create a new **project**.
4.  **Get API Keys:**
    *   Navigate to your project's settings.
    *   Go to the "API Keys" section.
    *   You will find your `Secret Key` and `Public Key`.
5.  **Update your environment file:**
    *   Copy the keys from the Langfuse interface.
    *   Paste them into your `memory_agents/.env` file. The `LANGFUSE_BASE_URL` should already be set correctly.

    ```
    # memory_agents/.env
    LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
    LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
    LANGFUSE_BASE_URL="http://localhost:3002"
    ```

#### Service ports

The `docker-compose.yml` in the root directory defines and orchestrates several services. Once `docker compose up -d` is executed, these services will be accessible via the following ports:

-   **Open WebUI**: A user interface for interacting with LLMs.
    -   Accessible at `http://localhost:8080`
-   **Graphiti MCP Server**: The core Graphiti memory service.
    -   Accessible at `http://localhost:8000` (as configured in `memory_agents/core/config.py`)
-   **Langfuse WebUI**: User interface for Langfuse tracing and analytics.
    -   Accessible at `http://localhost:3002`
-   **Langfuse Worker**: Backend worker for Langfuse (usually not directly accessed by the user).
    -   Exposed on port `3030` (internal to Docker network)
-   **ClickHouse**: Database for Langfuse (internal).
    -   Exposed on ports `8123`, `9000`, `9009` (internal)
-   **MinIO**: Object storage for Langfuse (internal).
    -   Exposed on ports `9000`, `9001` (internal)
-   **PostgreSQL**: Database for Langfuse (internal).
    -   Exposed on port `5432` (internal)
-   **Redis/Valkey**: Caching/message broker for Langfuse (internal).
    -   Exposed on port `6379` (internal)

## How to run

### Step 1: Run dependent services

In the project's root directory, run `docker compose up` to start all services required for the agents to run. This includes Langfuse for tracing and OpenWebUI.

```bash
# Make sure to execute this in the root directory. Use -d to run in detached mode.
docker compose up -d
```

### Step 2: Run a memory agent

All Python scripts must be executed from within the `memory_agents` directory.

To run a script, first navigate to the `memory_agents` directory. You can then either use `uv run` to execute it within the managed virtual environment, or activate the environment yourself before running it.

```bash
cd memory_agents
```

#### Option A: Using `uv run` (recommended)
This is the simplest way to run a script.

```bash
uv run pytest tests/agent_query_test.py::test_query_graphiti_agent
```

#### Option B: Activating the virtual environment
If you prefer to work inside the virtual environment's shell:

```bash
# Activate the virtual environment (on macOS/Linux)
source .venv/bin/activate

# Now you can run python scripts directly
pytest tests/agent_query_test.py::test_query_graphiti_agent

# Deactivate when you're done
deactivate
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
├── langfuse                            # Langfuse tracing and analytics
│   ├── .env.example                    # Langfuse environment file
│   └── docker-compose.yml              # Docker compose file for running Langfuse
├── memory_agents                       # The main application directory
│   ├── core
│   │   ├── agents
│   │   │   ├── baseline.py             # A simple agent with in-memory message history
|   |   |   ├── baseline_vdb.py         # A simple agent  that uses a vector database for memory
│   │   │   ├── graphiti.py             # An agent that uses graphiti for memory
│   │   │   └── graphiti_vdb.py         # An agent that uses graphiti and a vector database for memory
│   │   ├── config.py                   # Configuration for the agents
│   │   ├── middleware
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
├── docker-compose-workflow.yml         # Docker compose file for running services (light version for running in a Github workflow)
├── docker-compose.yml                  # Docker compose file for running services
├── LICENSE                             # Project license
└── README.md                           # Main README for the project
```

## Python formatting and code quality checks

```
cd memory_agents
uv run ruff format .
```

```
cd memory_agents
uv run ruff check .
uv run ruff check --fix
```

## Run PyTest tests

```
cd memory_agents
uv run pytest .
```

## Benchmarking

For detailed instructions on running benchmarks, please refer to the [LongMemEval Execution Guide](memory_agents/longmemeval/README.md).

## Troubleshooting

Agent initialization fails:

````
n3.12/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    |     raise mapped_exc(message) from exc
    | httpx.ConnectError: All connection attempts failed
````

Solution: Start Docker Compose backend services as instructed.