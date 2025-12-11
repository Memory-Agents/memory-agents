"""Tests for agent query functionality.

This module contains tests to verify that different agent types can
process queries and return valid responses. Each test sends a simple
"Hello World!" message to an agent and verifies that a non-null string
response is returned.

The tests cover:
- BaselineAgent query processing
- GraphitiAgent query processing
- BaselineVDBAgent query processing
- GraphitiVDBAgent query processing

These are basic smoke tests to ensure agents can handle simple queries
without errors and return properly formatted responses.
"""

from dotenv import load_dotenv

from memory_agents.core.run_agent import run_agent

load_dotenv()

import pytest


@pytest.mark.asyncio
async def test_query_baseline_agent():
    """Test BaselineAgent query processing.

    Sends a simple "Hello World!" message to a BaselineAgent and verifies
    that a valid string response is returned.

    Args:
        None: Uses hardcoded test message and thread ID.

    Returns:
        None: Raises AssertionError if query processing fails.

    Raises:
        AssertionError: If response is None or not a string.
    """
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    thread_id = "1"
    response = await run_agent(baseline_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_graphiti_agent():
    """Test GraphitiAgent query processing.

    Sends a simple "Hello World!" message to a GraphitiAgent and verifies
    that a valid string response is returned.

    Args:
        None: Uses hardcoded test message and thread ID.

    Returns:
        None: Raises AssertionError if query processing fails.

    Raises:
        AssertionError: If response is None or not a string.
    """
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphiti_agent = await GraphitiAgent.create()
    thread_id = "1"
    response = await run_agent(graphiti_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_baseline_vdb_agent():
    """Test BaselineVDBAgent query processing.

    Sends a simple "Hello World!" message to a BaselineVDBAgent and verifies
    that a valid string response is returned.

    Args:
        None: Uses hardcoded test message and thread ID.

    Returns:
        None: Raises AssertionError if query processing fails.

    Raises:
        AssertionError: If response is None or not a string.
    """
    from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent

    baseline_vdb_agent = BaselineVDBAgent()
    thread_id = "1"
    response = await run_agent(baseline_vdb_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_graphiti_vbd_agent():
    """Test GraphitiVDBAgent query processing.

    Sends a simple "Hello World!" message to a GraphitiVDBAgent and verifies
    that a valid string response is returned.

    Args:
        None: Uses hardcoded test message and thread ID.

    Returns:
        None: Raises AssertionError if query processing fails.

    Raises:
        AssertionError: If response is None or not a string.
    """
    from memory_agents.core.agents.graphiti_vdb import GraphitiVDBAgent

    graphiti_vbd_agent = await GraphitiVDBAgent.create()
    thread_id = "1"
    response = await run_agent(graphiti_vbd_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)
