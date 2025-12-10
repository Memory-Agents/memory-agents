from enum import Enum
from langchain.agents.middleware import AgentState
from langchain_core.messages.utils import AnyMessage


class MessageType(Enum):
    HUMAN = "human"
    AI = "ai"


def get_latest_message_from_agent_state(state: AgentState, type: MessageType):
    """Gets the latest message of a specific type from the agent state.

    Args:
        state: The agent state containing messages.
        type: The message type to retrieve (human or assistant).

    Returns:
        The latest message of the specified type, or None if not found.
    """
    messages: list[AnyMessage] = state["messages"]

    message = None
    for message in reversed(messages):
        if hasattr(message, "type") and message.type == type:
            message = message
            break

    return message


def insert_thread_id_in_state(state: AgentState, thread_id: str):
    """
    TODO: Implement
    """
    pass


def get_thread_id_in_state(state: AgentState) -> str:
    """
    TODO: Implement
    """
    return "TODO"
