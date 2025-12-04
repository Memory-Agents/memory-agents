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

    graphiti_agent = await GraphitiAgent.create()
    assert graphiti_agent.agent is not None


@pytest.mark.asyncio
async def test_init_baseline_vdb_agent_code():
    from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent

    baseline_vdb_agent = BaselineVDBAgent()
    assert baseline_vdb_agent.agent is not None


@pytest.mark.asyncio
async def test_init_graphiti_vdb_agent_code():
    from memory_agents.core.agents.graphiti_vdb import GraphitiChromaDBAgent

    graphiti_vdb_agent = await GraphitiChromaDBAgent.create()
    assert graphiti_vdb_agent.agent is not None
