from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from memory_agents.core.middleware.graphiti_retrieval_middleware_utils import (
    GraphitiRetrievalMiddlewareUtils,
)


class GraphitiRetrievalMiddleware(AgentMiddleware, GraphitiRetrievalMiddlewareUtils):
    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        super().__init__()
        self.graphiti_tools = graphiti_tools

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        nodes, memory_facts = self._retrieve_graphiti_with_user_message(state)
        retrieval_context = self._build_graphiti_augmentation_context_message(
            (nodes, memory_facts)
        )

        system_message = SystemMessage(content=retrieval_context)
        state["messages"].append(system_message)
        return None
