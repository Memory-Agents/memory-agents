from dotenv import load_dotenv

load_dotenv()

import pytest


def test_init_baseline_agent_code():
    from memory_agents.core.agents.baseline import agent

    assert agent is not None


def test_init_graphiti_agent_code():
    from memory_agents.core.agents.graphiti import create_agent
    import asyncio

    async def init_agent():
        agent = await create_agent()
        return agent

    agent = asyncio.run(init_agent())
    assert agent is not None
