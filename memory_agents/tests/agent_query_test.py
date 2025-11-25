from dotenv import load_dotenv

from memory_agents.core.run_agent import run_agent

load_dotenv()

import pytest


@pytest.mark.asyncio
async def test_query_baseline_agent():
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    response = await run_agent(baseline_agent.agent, "Hello World!")
    assert response is not None
    assert isinstance(response, str)


@pytest.mark.asyncio
async def test_init_graphiti_agent_code():
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphit_agent = await GraphitiAgent.create()
    response = await run_agent(graphit_agent.agent, "Hello World!")
    assert response is not None
    assert isinstance(response, str)
