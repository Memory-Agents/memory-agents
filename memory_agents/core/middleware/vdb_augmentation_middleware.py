from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from memory_agents.core.chroma_db_manager import ChromaDBManager

from langgraph.runtime import Runtime

from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)
from memory_agents.core.utils.message_conversion_utils import (
    ensure_message_content_is_str,
)


class VDBAugmentationMiddleware(AgentMiddleware):
    """Middleware for augmenting conversations with vector database storage.

    This middleware captures conversation turns (user and AI messages) and
    stores them in ChromaDB for future retrieval. It maintains context
    across conversations by storing each interaction with metadata.

    Attributes:
        chroma_manager (ChromaDBManager): Manager for ChromaDB operations.
        pending_user_message: The most recent user message waiting to be
            processed for storage.
    """

    def __init__(self, chroma_manager: ChromaDBManager):
        """Initialize the VDB augmentation middleware.

        Args:
            chroma_manager (ChromaDBManager): Manager for ChromaDB vector storage operations.
        """
        super().__init__()
        self.chroma_manager = chroma_manager
        self.pending_user_message = None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Extract and store the latest user message before model processing.

        This method captures the most recent human message from the agent state
        and stores it for later processing in the after_model hook.

        Args:
            state (AgentState): The current agent state containing messages.
            runtime (Runtime): The LangChain runtime instance.

        Returns:
            dict[str, Any] | None: Always returns None as no state modifications
                are needed.
        """
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)

        self.pending_user_message = message
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Store conversation turn in vector database after model response.

        This method pairs the latest AI response with the previously stored
        user message and stores the complete conversation turn in ChromaDB
        with relevant metadata for future retrieval.

        Args:
            state (AgentState): The current agent state containing messages.
            runtime (Runtime): The LangChain runtime instance.

        Returns:
            dict[str, Any] | None: Always returns None as no state modifications
                are needed.

        Raises:
            ValueError: If the user message could not be retrieved.
        """
        ai_message_type = MessageType.AI
        ai_message = get_latest_message_from_agent_state(state, ai_message_type)
        user_message = self.pending_user_message

        thread_id = get_thread_id_in_state(state)

        if not user_message:
            raise ValueError("Could not retrieve user message.")

        user_message_content = ensure_message_content_is_str(user_message.content)
        ai_message_content = ensure_message_content_is_str(ai_message.content)

        self.chroma_manager.add_conversation_turn(
            user_message=user_message_content,
            ai_message=ai_message_content,
            metadata={
                "thread_id": thread_id,
            },
        )
        return None
