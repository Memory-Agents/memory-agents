from typing import Any, Self
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model, AgentState, AgentMiddleware
from langgraph.runtime import Runtime
from langchain_mcp_adapters.client import MultiServerMCPClient

from memory_agents.core.config import GRAPHITI_MCP_URL


class GraphitiAgentMiddleware(AgentMiddleware):
    def insert_user_message_into_graphiti(
        state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        user_message = state.get_latest_user_message()
        if user_message:
            runtime.call_tool(
                "add_episode",
                {"message": user_message.content},
            )
        return None


class GraphitiAgent:
    def __init__(self):
        self.agent = None

    @classmethod
    async def create(cls) -> Self:
        self = cls()
        graphiti_tools = await self._get_graphiti_mcp_tools()
        self.agent = create_agent(
            model="gpt-4o-mini",
            system_prompt="You are a memory agent that helps the user to solve tasks.",
            checkpointer=InMemorySaver(),
            tools=graphiti_tools,
            middleware=[GraphitiAgentMiddleware()],
        )
        return self

    async def _get_graphiti_mcp_tools(self) -> Any:
        client = MultiServerMCPClient(
            {
                "graphiti": {
                    "transport": "streamable_http",  # HTTP-based remote server
                    "url": GRAPHITI_MCP_URL,
                }
            }
        )

        return await client.get_tools()
