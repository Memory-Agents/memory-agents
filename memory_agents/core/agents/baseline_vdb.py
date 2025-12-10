from typing import Any, Dict, List
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent
from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.config import (
    BASELINE_CHROMADB_DIR,
    BASELINE_MEMORY_PROMPT,
    BASELINE_MODEL_NAME,
)
from memory_agents.core.middleware.vdb_augmentation_middleware import (
    VDBAugmentationMiddleware,
)
from memory_agents.core.middleware.vdb_retrieval_middleware import (
    VDBRetrievalMiddleware,
)


class BaselineVDBAgent(ClearableAgent):
    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR) -> None:
        """Initialize the baseline VDB agent.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
        """
        self.chroma_manager = ChromaDBManager(persist_directory)

        agent: Any = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_MEMORY_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                VDBAugmentationMiddleware(self.chroma_manager),
                VDBRetrievalMiddleware(self.chroma_manager),
            ],
        )
        self.agent = agent

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
        """Clear all stored conversations from ChromaDB."""
        self.chroma_manager.clear_collection()
