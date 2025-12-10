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
    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        ThreadedSyncRunner.__init__(self)
        self.pending_user_message: AnyMessage | None = None
        self.graphiti_tools = graphiti_tools

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
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
        if not isinstance(ai_message, AIMessage):
            raise ValueError("AI message is not of the right type AIMessage")

        self._run_async_task(
            self._graphiti_augmentation(user_message, ai_message, thread_id)
        )
        time.sleep(10)
        return None


    async def _graphiti_augmentation(self, user_message, ai_message, thread_id):
        await self.graphiti_tools["add_memory"].ainvoke(
            {
                "name": "User Message",
                "episode_body": user_message.content,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        await self.graphiti_tools["add_memory"].ainvoke(
            {
                "name": "AI Message",
                "episode_body": ai_message.content,
                "group_id": thread_id,
                "sync": "True",
            }
        )
