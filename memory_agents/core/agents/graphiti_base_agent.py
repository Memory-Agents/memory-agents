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
        is_read_only=True,
        write_tools=[
            "add_memory",
            "delete_episode",
            "delete_entity_edge",
            "clear_graph",
        ],
    ) -> Any:
        """Get Graphiti MCP tools with optional filtering.

        Retrieves available tools from the Graphiti MCP client and filters them
        based on the is_read_only parameter. When is_read_only is True, excludes
        write tools to prevent unintended data modifications.

        Args:
            is_read_only: If True, excludes write tools from the parameter list.
            write_tools: List of tool names to exclude when is_read_only is True.

        Returns:
            Dictionary mapping tool names to their corresponding tool objects.
        """
        excluded_tools = []
        if is_read_only:
            excluded_tools = write_tools

        client = self._get_graphiti_client()

        tools = await client.get_tools()

        filtered_tools = {
            tool.name: tool for tool in tools if tool.name not in excluded_tools
        }
        return filtered_tools

    async def clear_graph(self, group_ids: list[str] | None = None) -> None:
        """Clear all data from the graph for specified group IDs.

        Args:
            group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
        """
        tools = await self._get_graphiti_mcp_tools()
        await tools["clear_graph"].ainvoke({"group_ids": group_ids})
