"""Tests for agent initialization functionality.

This module contains tests to verify that all agent types can be properly
initialized and have their underlying agent objects created successfully.

The tests cover:
- BaselineAgent initialization
- GraphitiAgent initialization
- BaselineVDBAgent initialization
- GraphitiVDBAgent initialization

Each test ensures that the agent object is not None after initialization,
indicating successful creation of the underlying agent infrastructure.
"""

from dotenv import load_dotenv

load_dotenv()
import pytest


def test_init_baseline_agent_code():
    """Test BaselineAgent initialization.

    Verifies that a BaselineAgent can be instantiated and that its
    underlying agent attribute is properly initialized.

    Returns:
        None: Raises AssertionError if agent initialization fails.

    Raises:
        AssertionError: If the baseline_agent.agent is None.
    """
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    assert baseline_agent.agent is not None


@pytest.mark.asyncio
async def test_init_graphiti_agent_code():
    """Test GraphitiAgent initialization.

    Verifies that a GraphitiAgent can be created asynchronously and that
    its underlying agent attribute is properly initialized.

    Returns:
        None: Raises AssertionError if agent creation fails.

    Raises:
        AssertionError: If the graphiti_agent.agent is None.
    """
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphiti_agent = await GraphitiAgent.create()
    assert graphiti_agent.agent is not None


@pytest.mark.asyncio
async def test_init_baseline_vdb_agent_code():
    """Test BaselineVDBAgent initialization.

    Verifies that a BaselineVDBAgent can be instantiated and that its
    underlying agent attribute is properly initialized.

    Returns:
        None: Raises AssertionError if agent initialization fails.

    Raises:
        AssertionError: If the baseline_vdb_agent.agent is None.
    """
    from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent

    baseline_vdb_agent = BaselineVDBAgent()
    assert baseline_vdb_agent.agent is not None


@pytest.mark.asyncio
async def test_init_graphiti_vdb_agent_code():
    """Test GraphitiVDBAgent initialization.

    Verifies that a GraphitiVDBAgent can be created asynchronously and that
    its underlying agent attribute is properly initialized.

    Returns:
        None: Raises AssertionError if agent creation fails.

    Raises:
        AssertionError: If the graphiti_vdb_agent.agent is None.
    """
    from memory_agents.core.agents.graphiti_vdb import GraphitiVDBAgent

    graphiti_vdb_agent = await GraphitiVDBAgent.create()
    assert graphiti_vdb_agent.agent is not None
