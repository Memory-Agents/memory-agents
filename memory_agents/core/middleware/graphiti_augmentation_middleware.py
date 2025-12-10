from langchain.agents.middleware import AgentMiddleware
from typing import Any
from langchain.agents.middleware import (
    AgentState,
)
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from memory_agents.core.utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)


class GraphitiAugmentationMiddleware(AgentMiddleware):
    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        super().__init__()
        self.pending_user_message = None
        self.graphiti_tools = graphiti_tools

    def before_model(self, state: AgentState, _: Runtime) -> dict[str, Any] | None:
        human_message_type = MessageType.HUMAN
        self.pending_user_message = get_latest_message_from_agent_state(
            state, human_message_type
        )
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        ai_message_type = MessageType.AI
        ai_message = get_latest_message_from_agent_state(state, ai_message_type)
        user_message = self.pending_user_message

        thread_id = get_thread_id_in_state(state)
        if not isinstance(user_message, HumanMessage):
            raise ValueError("User message is not of the right type UserMessage")
        if not isinstance(ai_message_type, AIMessage):
            raise ValueError("AI message is not of the right type AIMessage")
        self.graphiti_tools["add_memory"].invoke(
            {
                "name": "User Message",
                "episode_body": user_message.content,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        self.graphiti_tools["add_memory"].invoke(
            {
                "name": "AI Message",
                "episode_body": ai_message.content,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        return None
