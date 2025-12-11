from enum import Enum
from langchain.agents.middleware import AgentState
from langchain_core.messages.utils import AnyMessage


class MessageType(Enum):
    HUMAN = "human"
    AI = "ai"


def get_latest_message_from_agent_state(
    state: AgentState, type: MessageType
) -> AnyMessage:
    """Gets the latest message of a specific type from the agent state.

    Args:
        state: The agent state containing messages.
        type: The message type to retrieve (human or assistant).

    Returns:
        The latest message of the specified type, or None if not found.
    """
    messages: list[AnyMessage] = state["messages"]

    output_message = None
    for message in reversed(messages):
        if hasattr(message, "type") and message.type == type.value:
            output_message = message
            break

    if not output_message:
        raise ValueError("Message could not be found according to given type")

    return output_message


def insert_thread_id_in_state(state: AgentState, thread_id: str):
    """
    TODO: Implement
    """
    pass


def get_thread_id_in_state(state: AgentState) -> str:
    """
    TODO: Implement
    """
    return "1"
