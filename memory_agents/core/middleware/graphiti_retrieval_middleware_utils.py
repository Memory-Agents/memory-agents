from abc import ABC
from typing import Any, Tuple
from langchain.agents.middleware import AgentState

from memory_agents.core.utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)


class GraphitiRetrievalMiddlewareUtils(ABC):
    graphiti_tools: Any

    def _retrieve_graphiti_with_user_message(
        self, state: AgentState
    ) -> Tuple[str, str]:
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)
        thread_id = get_thread_id_in_state(state)

        graphiti_query = message.content

        nodes = self.graphiti_tools["search_nodes"].invoke(
            {
                "query": graphiti_query,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        memory_facts = self.graphiti_tools["search_memory_facts"].invoke(
            {
                "query": graphiti_query,
                "group_id": thread_id,
                "sync": "True",
            }
        )
        return nodes, memory_facts

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
