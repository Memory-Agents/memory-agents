# -*- coding: utf-8 -*-
"""Baseline memory agent implementation.

This module provides a basic memory agent implementation without persistent storage.
The agent uses in-memory checkpointing and serves as a foundation for more advanced
memory agents with vector database or knowledge graph integration.

Example:
    Creating and using a baseline agent:

    >>> from memory_agents.core.agents.baseline import BaselineAgent
    >>> agent = BaselineAgent()
    >>> # Agent is ready for use with basic memory functionality

"""

from typing import Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from memory_agents.config import BASELINE_MODEL_NAME
from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent


class BaselineAgent(ClearableAgent):
    """A baseline memory agent with in-memory checkpointing.

    This agent provides basic memory functionality using LangChain's create_agent
    with an in-memory checkpointer. It does not persist conversations across
    sessions and serves as a simple reference implementation.

    Attributes:
        agent: The underlying LangChain agent instance.

    Example:
        >>> agent = BaselineAgent()
        >>> # Use agent for conversation
        >>> await agent.clear_agent_memory()  # Clear in-memory state
    """

    def __init__(self) -> None:
        """Initialize the baseline agent.

        Creates a LangChain agent with a basic system prompt and in-memory
        checkpointer for conversation state management.
        """
        agent: Any = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt="You are a memory agent that helps the user to solve tasks.",
            checkpointer=InMemorySaver(),
        )
        self.agent: Any = agent

    async def clear_agent_memory(self):
        """Clear the agent's in-memory state.

        Since this agent uses only in-memory checkpointing, this method
        effectively resets the conversation state by doing nothing.
        The in-memory checkpointer will be cleared when the agent instance
        is destroyed or a new checkpointer is created.
        """
        pass
