from typing import Any, Self, List, Dict
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent
from memory_agents.core.agents.graphiti_base_agent import GraphitiBaseAgent
from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.config import (
    BASELINE_MEMORY_PROMPT,
    BASELINE_MODEL_NAME,
    GRAPHITI_VDB_CHROMADB_DIR,
)
from memory_agents.core.middleware.graphiti_augmentation_middleware import (
    GraphitiAugmentationMiddleware,
)
from memory_agents.core.middleware.graphiti_vdb_retrieval_middleware import (
    GraphitiVDBRetrievalMiddleware,
)
from memory_agents.core.middleware.vdb_augmentation_middleware import (
    VDBAugmentationMiddleware,
)


class GraphitiVDBAgent(GraphitiBaseAgent, ClearableAgent):
    def __init__(self):
        """Initialize the hybrid agent."""
        self.agent: Any = None
        self.chroma_manager: ChromaDBManager = None

    @classmethod
    async def create(cls, persist_directory: str = GRAPHITI_VDB_CHROMADB_DIR) -> Self:
        self = cls()

        self.chroma_manager = ChromaDBManager(persist_directory)

        graphiti_tools_all = await self._get_graphiti_mcp_tools(is_read_only=False)
        self.agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_MEMORY_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                GraphitiVDBRetrievalMiddleware(graphiti_tools_all),
                GraphitiAugmentationMiddleware(graphiti_tools_all),
                VDBAugmentationMiddleware(self.chroma_manager),
            ],
        )
        return self

    def get_chromadb_stats(self) -> Dict[str, int]:
        """Returns ChromaDB statistics.

        Returns:
            Dictionary containing database statistics like total conversation turns.
        """
        if not self.chroma_manager:
            return {}

        return {
            "total_conversation_turns": self.chroma_manager.conversation_collection.count()
        }

    def search_past_conversations(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Allows manual search in past conversations.

        Args:
            query: Search query for finding relevant conversations.
            n_results: Maximum number of results to return.

        Returns:
            List of conversation dictionaries matching the search query.
        """
        if not self.chroma_manager:
            return []

        return self.chroma_manager.search_conversations(query, n_results)

    async def clear_agent_memory(self):
        """Clear all stored data from both ChromaDB and Graphiti."""
        self.chroma_manager.clear_collection()
        await self.clear_graph()
