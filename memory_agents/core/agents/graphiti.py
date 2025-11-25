from typing import Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime
from langchain_mcp_adapters.client import MultiServerMCPClient

from core.config import GRAPHITI_MCP_URL


async def get_graphiti_mcp_tools() -> Any:
    client = MultiServerMCPClient(
        {
            "graphiti": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": GRAPHITI_MCP_URL,
            }
        }
    )

    return await client.get_tools()


@before_model(tools=["graphiti_add_user_message"])
def insert_user_message_into_graphiti(
    state: AgentState, runtime: Runtime
) -> dict[str, Any] | None:
    user_message = state.get_latest_user_message()
    if user_message:
        runtime.call_tool(
            "graphiti_add_user_message",
            {"message": user_message.content},
        )
    return None


async def create_agent() -> Any:
    graphiti_tools = await get_graphiti_mcp_tools()
    agent = create_agent(
        model="gpt-4o-mini",
        system_prompt="You are a memory agent that helps the user to solve tasks.",
        checkpointer=InMemorySaver(),
    )
