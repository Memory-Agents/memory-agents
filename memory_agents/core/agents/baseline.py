from typing import Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


class BaselineAgent:
    def __init__(self) -> Any:
        agent = create_agent(
            model="gpt-4o-mini",
            system_prompt="You are a memory agent that helps the user to solve tasks.",
            checkpointer=InMemorySaver(),
        )
        self.agent = agent
