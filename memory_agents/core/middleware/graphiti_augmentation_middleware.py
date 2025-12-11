import time
from langchain.agents.middleware import AgentMiddleware
from typing import Any
from langchain.agents.middleware import (
    AgentState,
)
from langchain_core.messages.utils import AnyMessage
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)
from memory_agents.core.utils.sync_runner import ThreadedSyncRunner


class GraphitiAugmentationMiddleware(AgentMiddleware, ThreadedSyncRunner):
    """Middleware for augmenting conversations with Graphiti memory storage.

    This middleware captures user and AI messages after each interaction
    and stores them as episodic memories in Graphiti for future retrieval.
    Creates a thread to run async function to completion, due to async interfaces from dependencies.

    Attributes:
        pending_user_message (AnyMessage | None): The most recent user message
            waiting to be processed for memory storage.
        graphiti_tools (dict[str, BaseTool]): Dictionary containing Graphiti tools
            for memory operations, specifically the 'add_memory' tool.
    """

    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        """Initialize the Graphiti augmentation middleware.

        Args:
            graphiti_tools (dict[str, BaseTool]): Dictionary containing Graphiti tools
                for memory operations. Must include an 'add_memory' tool.
        """
        ThreadedSyncRunner.__init__(self)
        self.pending_user_message: AnyMessage | None = None
        self.graphiti_tools = graphiti_tools

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
        self.pending_user_message = get_latest_message_from_agent_state(
            state, human_message_type
        )
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Process and store conversation turn in Graphiti after model response.

        This method extracts the latest AI response and pairs it with the
        previously stored user message to create episodic memories in Graphiti.
        The augmentation runs asynchronously with a delay to ensure proper processing.

        Args:
            state (AgentState): The current agent state containing messages.
            runtime (Runtime): The LangChain runtime instance.

        Returns:
            dict[str, Any] | None: Always returns None as no state modifications
                are needed.

        Raises:
            ValueError: If the user message is not a HumanMessage instance.
            ValueError: If the AI message is not an AIMessage instance.
        """
        ai_message_type = MessageType.AI
        ai_message = get_latest_message_from_agent_state(state, ai_message_type)
        user_message = self.pending_user_message

        thread_id = get_thread_id_in_state(state)
        if not isinstance(user_message, HumanMessage):
            raise ValueError("User message is not of the right type UserMessage")
        if not isinstance(ai_message, AIMessage):
            raise ValueError("AI message is not of the right type AIMessage")

        self._run_async_task(
            self._graphiti_augmentation(user_message, ai_message, thread_id)
        )
        time.sleep(10)
        return None

    async def _graphiti_augmentation(
        self, user_message: HumanMessage, ai_message: AIMessage, thread_id: str
    ):
        """Asynchronously store user and AI messages as episodic memories.

        This method creates two separate memory entries in Graphiti - one for
        the user message and one for the AI response. These are stored as
        episodic memories that can be retrieved in future conversations.

        Args:
            user_message (HumanMessage): The user's message content to store.
            ai_message (AIMessage): The AI's response content to store.
            thread_id (str): The conversation thread identifier for context.
        """
        await self.graphiti_tools["add_memory"].ainvoke(
            {
                "name": "User Message",
                "episode_body": user_message.content,
            }
        )
        await self.graphiti_tools["add_memory"].ainvoke(
            {
                "name": "AI Message",
                "episode_body": ai_message.content,
            }
        )
