from typing import Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from memory_agents.config import BASELINE_MODEL_NAME
from memory_agents.core.agents.clearable_agent import ClearableAgent


class BaselineAgent(ClearableAgent):
    def __init__(self) -> Any:
        agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt="You are a memory agent that helps the user to solve tasks.",
            checkpointer=InMemorySaver(),
        )
        self.agent = agent

    async def clear_agent_memory(self):
        pass
