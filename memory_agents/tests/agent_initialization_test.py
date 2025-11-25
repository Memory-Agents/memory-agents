from dotenv import load_dotenv

load_dotenv()

import pytest


def test_init_baseline_agent_code():
    from memory_agents.core.agents.baseline import BaselineAgent

    baseline_agent = BaselineAgent()
    assert baseline_agent.agent is not None


@pytest.mark.asyncio
async def test_init_graphiti_agent_code():
    from memory_agents.core.agents.graphiti import GraphitiAgent

    graphit_agent = await GraphitiAgent.create()
    assert graphit_agent.agent is not None
