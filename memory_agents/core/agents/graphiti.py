# -*- coding: utf-8 -*-
"""Graphiti knowledge graph memory agent.

This module provides a memory agent implementation that integrates with Graphiti
knowledge graph through MCP (Model Context Protocol) tools. The agent uses
knowledge graph middleware for storing and retrieving structured information
about entities, relationships, and conversations.

Example:
    Creating and using a Graphiti agent:

    >>> from memory_agents.core.agents.graphiti import GraphitiAgent
    >>> agent = await GraphitiAgent.create()
    >>> # Agent is ready for use with knowledge graph memory

"""

from typing import Self
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent
from memory_agents.core.agents.graphiti_base_agent import GraphitiBaseAgent
from memory_agents.config import BASELINE_MODEL_NAME

from memory_agents.core.config import BASELINE_MEMORY_PROMPT
from memory_agents.core.middleware.graphiti_augmentation_middleware import (
    GraphitiAugmentationMiddleware,
)
from memory_agents.core.middleware.graphiti_retrieval_middleware import (
    GraphitiRetrievalMiddleware,
)


class GraphitiAgent(GraphitiBaseAgent, ClearableAgent):
    """A memory agent with Graphiti knowledge graph integration.

    This agent combines the functionality of GraphitiBaseAgent and ClearableAgent
    to provide knowledge graph-based memory capabilities. It uses Graphiti MCP tools
    for storing and retrieving structured information about entities, relationships,
    and conversation context.

    Attributes:
        agent: The underlying LangChain agent instance with Graphiti middleware.

    Example:
        >>> agent = await GraphitiAgent.create()
        >>> # Agent is ready for use with knowledge graph memory
        >>> await agent.clear_agent_memory()  # Clear all graph data
    """

    def __init__(self):
        """Initialize the Graphiti agent.

        Creates a new instance with agent attribute set to None.
        The actual agent creation happens in the async create() method
        to allow for async operations during initialization.
        """
        self.agent = None

    @classmethod
    async def create(cls) -> Self:
        """Create and initialize a Graphiti agent instance.

        This class method handles the async initialization process, including
        retrieving Graphiti MCP tools and setting up the LangChain agent with
        appropriate middleware for knowledge graph operations.

        Returns:
            An initialized GraphitiAgent instance ready for use.

        Note:
            This method must be called instead of __init__ to properly
            initialize the agent with async operations.
        """
        self = cls()
        graphiti_tools_all = await self._get_graphiti_mcp_tools(is_read_only=False)
        self.agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_MEMORY_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                GraphitiAugmentationMiddleware(graphiti_tools_all),
                GraphitiRetrievalMiddleware(graphiti_tools_all),
            ],
        )
        return self

    async def clear_agent_memory(self):
        """Clear all data from the Graphiti knowledge graph.

        Removes all entities, relationships, and episodes from the knowledge
        graph, effectively resetting the agent's structured memory state.
        This operation is irreversible and will delete all stored knowledge.

        Note:
            This method calls the inherited clear_graph() method from
            GraphitiBaseAgent to perform the actual clearing operation.
        """
        await self.clear_graph()
