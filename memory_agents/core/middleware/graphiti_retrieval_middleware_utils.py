from typing import Any, Tuple
from langchain.agents.middleware import AgentState
import logging

from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
    get_thread_id_in_state,
)
from memory_agents.core.utils.message_conversion_utils import (
    ensure_message_content_is_str,
)
from memory_agents.core.utils.sync_runner import ThreadedSyncRunner


class GraphitiRetrievalMiddlewareUtils(ThreadedSyncRunner):
    """Utility class providing Graphiti memory retrieval functionality.

    This class contains helper methods for retrieving relevant memories from
    Graphiti based on user queries and building context messages for AI responses.
    Creates a thread to run async function to completion, due to async interfaces from dependencies.

    Attributes:
        graphiti_tools (Any): Graphiti tools for memory retrieval operations.
        logger: Logger instance for debugging and error reporting.
    """

    graphiti_tools: Any

    def __init__(self):
        """Initialize the Graphiti retrieval utilities.

        Sets up the logger and initializes the threaded sync runner functionality.
        """
        self.logger = logging.getLogger()
        ThreadedSyncRunner.__init__(self)

    def _retrieve_graphiti_with_user_message(
        self, state: AgentState
    ) -> Tuple[str, str]:
        """Retrieve relevant Graphiti memories based on the user's latest message.

        This method extracts the latest human message from the agent state,
        converts it to a string query, and searches Graphiti for relevant
        nodes and memory facts.

        Args:
            state (AgentState): The current agent state containing messages.

        Returns:
            Tuple[str, str]: A tuple containing (nodes, memory_facts) retrieved
                from Graphiti based on the user's query.
        """
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)
        thread_id = get_thread_id_in_state(state)

        graphiti_query = ensure_message_content_is_str(message.content)

        return self._run_async_task(self._graphiti_retrieval(graphiti_query, thread_id))

    def _build_graphiti_augmentation_context_message(
        self, nodes_and_memory_facts: Tuple[str, str]
    ) -> str:
        """Build a context message from retrieved Graphiti nodes and memory facts.

        This method formats the retrieved nodes and memory facts into a structured
        context message that can be injected into the AI's conversation to provide
        relevant background information.

        Args:
            nodes_and_memory_facts (Tuple[str, str]): A tuple containing (nodes, memory_facts)
                retrieved from Graphiti.

        Returns:
            str: A formatted context message containing the retrieved information
                with instructions for appropriate usage.
        """
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
        """Asynchronously search Graphiti for relevant nodes and memory facts.

        This method performs parallel searches in Graphiti for nodes and memory facts
        that match the given query, providing comprehensive context for the AI.

        Args:
            graphiti_query (str): The search query based on the user's message.
            thread_id (str): The conversation thread identifier for context.

        Returns:
            Tuple[str, str]: A tuple containing (nodes, memory_facts) retrieved
                from Graphiti based on the query.
        """
        nodes = await self.graphiti_tools["search_nodes"].ainvoke(
            {
                "query": graphiti_query,
            }
        )
        memory_facts = await self.graphiti_tools["search_memory_facts"].ainvoke(
            {
                "query": graphiti_query,
            }
        )
        return nodes, memory_facts
