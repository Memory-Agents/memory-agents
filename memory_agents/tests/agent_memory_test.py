from dotenv import load_dotenv
import pytest
import shutil
import os
from memory_agents.core.agents.baseline_vdb import (
    BASELINE_CHROMADB_SYSTEM_PROMPT,
    ChromaDBStorageMiddleware,
    RAGEnhancedAgentMiddleware,
)
from memory_agents.core.config import BASELINE_MODEL_NAME
from memory_agents.core.run_agent import run_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Define a secret that the agent should rememfber
SECRET_CODE = "My favorite color is blue."
SECRET_QUESTION = "What is my favorite color?"
SECRET_ANSWER = "blue"

# Define a unique directory for ChromaDB for this test run to avoid conflicts
TEST_CHROMADB_DIR = "./test_chroma_db"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown_chroma():
    # Setup: ensure the directory is clean before tests
    if os.path.exists(TEST_CHROMADB_DIR):
        shutil.rmtree(TEST_CHROMADB_DIR)
    os.makedirs(TEST_CHROMADB_DIR)

    yield

    # Teardown: clean up the directory after tests
    if os.path.exists(TEST_CHROMADB_DIR):
        shutil.rmtree(TEST_CHROMADB_DIR)


@pytest.mark.asyncio
async def test_memory_baseline_vdb_agent():
    """Test ChromaDB vector memory with RAG - uses middleware for automatic storage"""
    from memory_agents.core.agents.baseline_vdb import BaselineAgent

    baseline_vdb_agent = BaselineAgent(persist_directory=TEST_CHROMADB_DIR)

    # --- Conversation 1: Introduce the secret ---
    thread_id_1 = "memory_test_thread_1_baseline_vdb"

    # Store conversations via middleware
    await run_agent(baseline_vdb_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        baseline_vdb_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(baseline_vdb_agent.agent, "Tell me a joke.", thread_id_1)

    # Reset InMemory to test ChromaDB-only retrieval
    baseline_vdb_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=BASELINE_CHROMADB_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            RAGEnhancedAgentMiddleware(baseline_vdb_agent.chroma_manager),
            ChromaDBStorageMiddleware(baseline_vdb_agent.chroma_manager),
        ],
    )

    # --- Conversation 2: Test memory retrieval ---
    response = await run_agent(baseline_vdb_agent.agent, SECRET_QUESTION, thread_id_1)

    # Verify memory retrieval
    assert SECRET_ANSWER.lower() in response.lower()


@pytest.mark.asyncio
async def test_memory_graphiti_agent():
    from memory_agents.core.agents.graphiti import (
        GRAPHITI_SYSTEM_PROMPT,
        GraphitiAgent,
        GraphitiAgentMiddleware,
    )

    graphiti_agent = await GraphitiAgent.create()

    # --- Conversation 1: Introduce the secret ---
    thread_id_1 = "memory_test_thread_1_graphiti"

    # Store conversations via middleware
    await run_agent(graphiti_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        graphiti_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(graphiti_agent.agent, "Tell me a joke.", thread_id_1)

    # Reinitialize agent
    graphiti_tools = await graphiti_agent._get_graphiti_mcp_tools()
    graphiti_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=GRAPHITI_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        tools=list(graphiti_tools.values()),
        middleware=[GraphitiAgentMiddleware(graphiti_tools)],
    )

    # --- Conversation 2: Test memory retrieval ---
    response = await run_agent(graphiti_agent.agent, SECRET_QUESTION, thread_id_1)

    # Assert that the agent remembers the secret
    assert SECRET_ANSWER.lower() in response.lower(), (
        f"Expected '{SECRET_ANSWER}' but got: {response}"
    )


@pytest.mark.asyncio
async def test_memory_graphiti_vdb_agent():
    from memory_agents.core.agents.graphiti_vdb import (
        GRAPHITI_CHROMADB_SYSTEM_PROMPT,
        GraphitiChromaDBAgent,
        RAGEnhancedAgentMiddleware,
        GraphitiChromaDBStorageMiddleware,
    )

    # Use a subdirectory for this agent's ChromaDB
    graphiti_vdb_chroma_dir = os.path.join(TEST_CHROMADB_DIR, "graphiti_vdb")

    graphiti_vdb_agent = await GraphitiChromaDBAgent.create(
        persist_directory=graphiti_vdb_chroma_dir
    )

    # --- Conversation 1: Introduce the secret ---
    thread_id_1 = "memory_test_thread_1_graphiti_vdb"

    # Store conversations via middleware
    await run_agent(graphiti_vdb_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        graphiti_vdb_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(graphiti_vdb_agent.agent, "Tell me a joke.", thread_id_1)

    # Reinitialize agent
    graphiti_tools = await graphiti_vdb_agent._get_graphiti_mcp_tools()
    graphiti_vdb_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=GRAPHITI_CHROMADB_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        tools=list(graphiti_tools.values()),
        middleware=[
            RAGEnhancedAgentMiddleware(graphiti_vdb_agent.chroma_manager),
            GraphitiChromaDBStorageMiddleware(graphiti_vdb_agent.chroma_manager, graphiti_tools),
        ],
    )

    # --- Conversation 2: Test memory retrieval ---
    response = await run_agent(graphiti_vdb_agent.agent, SECRET_QUESTION, thread_id_1)

    # Assert that the agent remembers the secret
    assert SECRET_ANSWER.lower() in response.lower(), (
        f"Expected '{SECRET_ANSWER}' but got: {response}"
    )
