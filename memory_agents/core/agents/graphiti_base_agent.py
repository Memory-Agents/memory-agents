# -*- coding: utf-8 -*-
"""Base class for Graphiti knowledge graph agents.

This module provides the foundational functionality for agents that integrate
with Graphiti knowledge graph through MCP (Model Context Protocol) tools.
It handles client creation, tool management, and basic graph operations.

Example:
    Using the base agent functionality:

    >>> from memory_agents.core.agents.graphiti_base_agent import GraphitiBaseAgent
    >>> class MyAgent(GraphitiBaseAgent):
    ...     async def get_tools(self):
    ...         return await self._get_graphiti_mcp_tools()

"""

from abc import ABC
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Any

from memory_agents.core.config import GRAPHITI_MCP_URL


class GraphitiBaseAgent(ABC):
    """Base class for Graphiti agents.

    This abstract base class provides common functionality for agents that
    interact with Graphiti knowledge graph. It handles MCP client creation,
    tool filtering, and basic graph operations that are shared across
    different Graphiti agent implementations.

    Example:
        >>> class MyGraphitiAgent(GraphitiBaseAgent):
        ...     async def setup_tools(self):
        ...         self.tools = await self._get_graphiti_mcp_tools()
    """

    def _get_graphiti_client(self, url: str = GRAPHITI_MCP_URL) -> MultiServerMCPClient:
        """Create a Graphiti MCP client.

        Args:
            url: The Graphiti MCP server URL. Defaults to configured URL.

        Returns:
            A configured MultiServerMCPClient instance for Graphiti operations.
        """
        client = MultiServerMCPClient(
            {
                "graphiti": {
                    "transport": "streamable_http",  # HTTP-based remote server
                    "url": GRAPHITI_MCP_URL,
                }
            }
        )
        return client

    async def _get_graphiti_mcp_tools(
        self,
        exclude: list[str] = [
            "add_memory",
            "delete_episode",
            "delete_entity_edge",
            "clear_graph",
        ],
    ) -> Any:
        """Get filtered Graphiti MCP tools.

        Retrieves available Graphiti tools and filters out memory-modifying
        tools to prevent unauthorized data modification. By default, excludes
        tools that can add, delete, or clear graph data.

        Args:
            exclude: List of tool names to exclude from the returned tools.
                    Defaults to memory-modifying tools.

        Returns:
            Dictionary of filtered Graphiti MCP tools keyed by tool name.
        """
        client = self._get_graphiti_client()

        tools = await client.get_tools()

        # Remove any tools that modify memory: * `add_episode` * `delete_episode` * `delete_entity_edge` * `clear_graph`
        filtered_tools = {tool.name: tool for tool in tools if tool.name not in exclude}
        return filtered_tools

    async def clear_graph(self, group_ids: list[str] | None = None) -> None:
        """Clear all data from the graph for specified group IDs.

        Args:
            group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
        """
        tools = await self._get_graphiti_mcp_tools(exclude=[])
        await tools["clear_graph"].ainvoke({"group_ids": group_ids})
