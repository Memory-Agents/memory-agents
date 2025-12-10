from typing import Any, Tuple
from langchain.agents.middleware import AgentState

from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)
from memory_agents.core.utils.sync_runner import ThreadedSyncRunner


class GraphitiRetrievalMiddlewareUtils(ThreadedSyncRunner):
    graphiti_tools: Any

    def __init__(self):
        ThreadedSyncRunner.__init__(self)

    def _retrieve_graphiti_with_user_message(
        self, state: AgentState
    ) -> Tuple[str, str]:
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)
        thread_id = get_thread_id_in_state(state)

        graphiti_query = message.content

        return self._run_async_task(self._graphiti_retrieval(graphiti_query, thread_id))

    def _build_augmentation_context_message(
        self, nodes_and_memory_facts: Tuple[str, str]
    ) -> str:
        nodes, memory_facts = nodes_and_memory_facts

        retrieval_context = f"""
            <retrieved_context>
            Retrieved nodes:
            {nodes}

            Retrieved memory facts:

            {memory_facts}
            </retrieved_context>

            IMPORTANT:
            Only use information from <retrieved_context> if it is clearly relevant to the user's query.
            If it is not relevant, IGNORE it entirely.
            """
        return retrieval_context

    async def _graphiti_retrieval(
        self, graphiti_query: str, thread_id: str
    ) -> Tuple[str, str]:
        nodes = await self.graphiti_tools["search_nodes"].ainvoke(
            {
                "query": graphiti_query,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        memory_facts = await self.graphiti_tools["search_memory_facts"].ainvoke(
            {
                "query": graphiti_query,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        return nodes, memory_facts
