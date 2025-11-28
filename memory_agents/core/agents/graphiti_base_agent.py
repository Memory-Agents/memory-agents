from abc import ABC
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Any

from memory_agents.core.config import GRAPHITI_MCP_URL


class GraphitiBaseAgent(ABC):
    """Base class for Graphiti agents."""

    async def _get_graphiti_mcp_tools(self) -> Any:
        client = MultiServerMCPClient(
            {
                "graphiti": {
                    "transport": "streamable_http",  # HTTP-based remote server
                    "url": GRAPHITI_MCP_URL,
                }
            }
        )

        tools = await client.get_tools()
        # Remove any tools that modify memory: * `add_episode` * `delete_episode` * `delete_entity_edge` * `clear_graph`
        filtered_tools = [
            tool
            for tool in tools
            if tool.name
            not in [
                "add_episode",
                "delete_episode",
                "delete_entity_edge",
                "clear_graph",
            ]
        ]
        return filtered_tools
