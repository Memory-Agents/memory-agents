from abc import ABC
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Any

from memory_agents.core.config import GRAPHITI_MCP_URL


def get_graphiti_client(url: str = GRAPHITI_MCP_URL) -> MultiServerMCPClient:
    client = MultiServerMCPClient(
        {
            "graphiti": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": GRAPHITI_MCP_URL,
            }
        }
    )
    return client


class GraphitiBaseAgent(ABC):
    """Base class for Graphiti agents."""

    async def _get_graphiti_mcp_tools(
        self,
        exclude: list[str] = [
            "delete_episode",
            "delete_entity_edge",
            "clear_graph",
        ],
    ) -> Any:
        client = get_graphiti_client()

        tools = await client.get_tools()

        # Remove any tools that modify memory: * `add_episode` * `delete_episode` * `delete_entity_edge` * `clear_graph`
        filtered_tools = {
            tool.name: tool for tool in tools if tool.name not in exclude
        }
        return filtered_tools

    async def clear_graph(self, group_ids: list[str] | None = None) -> None:
        """Clear all data from the graph for specified group IDs.

        Args:
            group_ids: Optional list of group IDs to clear. If not provided, clears the default group.
        """
        tools = await self._get_graphiti_mcp_tools(exclude=[])
        await tools["clear_graph"].ainvoke({"group_ids": group_ids})
