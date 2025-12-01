from abc import ABC
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import Any
import asyncio
from functools import wraps

from memory_agents.core.config import GRAPHITI_MCP_URL


class GraphitiBaseAgent(ABC):
    """Base class for Graphiti agents."""

    @staticmethod
    def make_tool_sync_compatible(tool):
        """Wraps an async tool to make it synchronously callable"""
        if hasattr(tool, 'coroutine') and tool.coroutine:
            # Tool is async, create a sync wrapper
            original_coroutine = tool.coroutine
            
            @wraps(original_coroutine)
            def sync_wrapper(*args, **kwargs):
                # Remove 'runtime' from kwargs if present, as it's handled by langgraph
                runtime = kwargs.pop('runtime', None)
                
                try:
                    # Try to get the current event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, create a new task
                        # This is a workaround for nested async calls
                        import nest_asyncio
                        nest_asyncio.apply()
                        return loop.run_until_complete(original_coroutine(*args, **kwargs))
                    else:
                        # Run in existing loop
                        return loop.run_until_complete(original_coroutine(*args, **kwargs))
                except RuntimeError:
                    # No event loop exists, create a new one
                    return asyncio.run(original_coroutine(*args, **kwargs))
            
            # Replace the tool's coroutine with sync wrapper
            tool.func = sync_wrapper
            tool.coroutine = None  # Mark as no longer async
        
        return tool

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
                "add_memory",
                "delete_episode",
                "delete_entity_edge",
                "clear_graph",
            ]
        ]
        
        # Make all tools sync-compatible to avoid "StructuredTool does not support sync invocation" error
        sync_compatible_tools = [self.make_tool_sync_compatible(tool) for tool in filtered_tools]
        
        return sync_compatible_tools
