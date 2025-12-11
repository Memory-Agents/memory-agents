from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from memory_agents.core.middleware.graphiti_retrieval_middleware_utils import (
    GraphitiRetrievalMiddlewareUtils,
)


class GraphitiRetrievalMiddleware(AgentMiddleware, GraphitiRetrievalMiddlewareUtils):
    """Middleware for retrieving relevant memories from Graphiti before model processing.

    This middleware searches Graphiti for relevant nodes and memory facts based on
    the user's latest message and injects this context as a system message to
    inform the AI's response.

    Attributes:
        graphiti_tools (dict[str, BaseTool]): Dictionary containing Graphiti tools
            for memory retrieval operations, including 'search_nodes' and
            'search_memory_facts' tools.
    """

    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        """Initialize the Graphiti retrieval middleware.

        Args:
            graphiti_tools (dict[str, BaseTool]): Dictionary containing Graphiti tools
                for memory retrieval operations. Must include 'search_nodes' and
                'search_memory_facts' tools.
        """
        super().__init__()
        self.graphiti_tools = graphiti_tools

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Retrieve relevant memories and inject them as context before model processing.

        This method searches Graphiti for relevant nodes and memory facts based on
        the user's latest message, builds a context message, and injects it as
        a system message to inform the AI's response.

        Args:
            state (AgentState): The current agent state containing messages.
            runtime (Runtime): The LangChain runtime instance.

        Returns:
            dict[str, Any] | None: Always returns None as state is modified in place.
        """
        nodes, memory_facts = self._retrieve_graphiti_with_user_message(state)
        retrieval_context = self._build_graphiti_augmentation_context_message(
            (nodes, memory_facts)
        )

        system_message = SystemMessage(content=retrieval_context)
        state["messages"].append(system_message)
        return None
