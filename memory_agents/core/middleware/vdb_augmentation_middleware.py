from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from memory_agents.core.chroma_db_manager import ChromaDBManager

from langgraph.runtime import Runtime

from memory_agents.core.utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)


class VDBAugmentationMiddleware(AgentMiddleware):
    def __init__(self, chroma_manager: ChromaDBManager):
        super().__init__()
        self.chroma_manager = chroma_manager
        self.pending_user_message = None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)

        self.pending_user_message = message
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        assistant_message_type = MessageType.AI
        assistant_message = get_latest_message_from_agent_state(
            state, assistant_message_type
        )
        user_message = self.pending_user_message

        thread_id = get_thread_id_in_state(state)

        if not user_message:
            raise ValueError("Could not retrieve user message.")

        self.chroma_manager.add_conversation_turn(
            user_message=user_message,
            assistant_message=assistant_message,
            metadata={
                "thread_id": thread_id,
            },
        )
        return None
