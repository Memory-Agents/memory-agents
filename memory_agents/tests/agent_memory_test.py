from dotenv import load_dotenv
import pytest
import shutil
import os
from memory_agents.core.config import BASELINE_MEMORY_PROMPT, BASELINE_MODEL_NAME
from memory_agents.core.middleware.graphiti_augmentation_middleware import (
    GraphitiAugmentationMiddleware,
)
from memory_agents.core.middleware.graphiti_retrieval_middleware import (
    GraphitiRetrievalMiddleware,
)
from memory_agents.core.middleware.graphiti_vdb_retrieval_middleware import (
    GraphitiVDBRetrievalMiddleware,
)
from memory_agents.core.middleware.vdb_augmentation_middleware import (
    VDBAugmentationMiddleware,
)
from memory_agents.core.middleware.vdb_retrieval_middleware import (
    VDBRetrievalMiddleware,
)
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
    from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent

    baseline_vdb_agent = BaselineVDBAgent(persist_directory=TEST_CHROMADB_DIR)

    # Conversation 1: Introduce the secret
    thread_id_1 = "memory_test_thread_1_baseline_vdb"

    await baseline_vdb_agent.clear_agent_memory()

    # Store conversations via middleware
    await run_agent(baseline_vdb_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        baseline_vdb_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(baseline_vdb_agent.agent, "Tell me a joke.", thread_id_1)

    # Reset InMemory to test memory-only retrieval
    baseline_vdb_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=BASELINE_MEMORY_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            VDBRetrievalMiddleware(baseline_vdb_agent.chroma_manager),
            VDBAugmentationMiddleware(baseline_vdb_agent.chroma_manager),
        ],
    )

    # Conversation 2: Test memory retrieval
    response = await run_agent(baseline_vdb_agent.agent, SECRET_QUESTION, thread_id_1)

    # Verify memory retrieval
    assert SECRET_ANSWER.lower() in response.lower()


@pytest.mark.asyncio
async def test_memory_graphiti_agent():
    """Test ChromaDB vector memory with RAG - uses middleware for automatic storage"""
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphiti_agent = await GraphitiAgent.create()

    # Conversation 1: Introduce the secret
    thread_id_1 = "memory_test_thread_1_baseline_vdb"

    await graphiti_agent.clear_agent_memory()

    # Store conversations via middleware
    await run_agent(graphiti_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        graphiti_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(graphiti_agent.agent, "Tell me a joke.", thread_id_1)

    # Reset InMemory to test memory-only retrieval
    graphiti_tools_all = await graphiti_agent._get_graphiti_mcp_tools(
        is_read_only=False
    )
    graphiti_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=BASELINE_MEMORY_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            GraphitiAugmentationMiddleware(graphiti_tools_all),
            GraphitiRetrievalMiddleware(graphiti_tools_all),
        ],
    )

    # Conversation 2: Test memory retrieval
    response = await run_agent(graphiti_agent.agent, SECRET_QUESTION, thread_id_1)

    # Verify memory retrieval
    assert SECRET_ANSWER.lower() in response.lower()


@pytest.mark.asyncio
async def test_memory_graphiti_vdb_agent():
    """Test ChromaDB vector memory with RAG - uses middleware for automatic storage"""
    from memory_agents.core.agents.graphiti_vdb import GraphitiVDBAgent

    graphiti_vdb_agent = await GraphitiVDBAgent.create(
        persist_directory=TEST_CHROMADB_DIR
    )

    # Conversation 1: Introduce the secret
    thread_id_1 = "memory_test_thread_1_baseline_vdb"

    await graphiti_vdb_agent.clear_agent_memory()

    # Store conversations via middleware
    await run_agent(graphiti_vdb_agent.agent, SECRET_CODE, thread_id_1)
    await run_agent(
        graphiti_vdb_agent.agent, "What is the weather like today?", thread_id_1
    )
    await run_agent(graphiti_vdb_agent.agent, "Tell me a joke.", thread_id_1)

    graphiti_tools_all = await graphiti_vdb_agent._get_graphiti_mcp_tools(
        is_read_only=False
    )
    graphiti_vdb_agent.agent = create_agent(
        model=BASELINE_MODEL_NAME,
        system_prompt=BASELINE_MEMORY_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            GraphitiVDBRetrievalMiddleware(
                graphiti_tools_all, graphiti_vdb_agent.chroma_manager
            ),
            GraphitiAugmentationMiddleware(graphiti_tools_all),
            VDBAugmentationMiddleware(graphiti_vdb_agent.chroma_manager),
        ],
    )

    # Conversation 2: Test memory retrieval
    response = await run_agent(graphiti_vdb_agent.agent, SECRET_QUESTION, thread_id_1)

    # Verify memory retrieval
    assert SECRET_ANSWER.lower() in response.lower()
