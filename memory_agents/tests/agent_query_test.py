from dotenv import load_dotenv

from memory_agents.core.run_agent import run_agent

load_dotenv()

import pytest


@pytest.mark.asyncio
async def test_query_baseline_agent():
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    thread_id = "1"
    response = await run_agent(baseline_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_graphiti_agent():
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphiti_agent = await GraphitiAgent.create()
    thread_id = "1"
    response = await run_agent(graphiti_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_baseline_vdb_agent():
    from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent

    baseline_vdb_agent = BaselineVDBAgent()
    thread_id = "1"
    response = await run_agent(baseline_vdb_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_query_graphiti_vbd_agent():
    from memory_agents.core.agents.graphiti_vdb import GraphitiChromaDBAgent

    graphiti_vbd_agent = await GraphitiChromaDBAgent.create()
    thread_id = "1"
    response = await run_agent(graphiti_vbd_agent.agent, "Hello World!", thread_id)
    assert response is not None
    assert isinstance(response, str)
