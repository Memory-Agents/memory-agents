# -*- coding: utf-8 -*-
"""Baseline memory agent with vector database integration.

This module provides a memory agent implementation that uses ChromaDB for
persistent conversation storage and retrieval. The agent combines LangChain's
create_agent with vector database middleware to enable long-term memory
capabilities across sessions.

Example:
    Creating and using a baseline VDB agent:

    >>> from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent
    >>> agent = BaselineVDBAgent("/path/to/db")
    >>> # Agent is ready for use with persistent memory
    >>> stats = agent.get_chromadb_stats()
    >>> results = agent.search_past_conversations("hello")

"""

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
    """A baseline memory agent with vector database integration.

    This agent extends the baseline functionality by integrating with ChromaDB
    for persistent conversation storage and retrieval. It uses middleware for
    both augmentation (storing conversations) and retrieval (searching past
    conversations) to provide long-term memory capabilities.

    Attributes:
        agent: The underlying LangChain agent instance with VDB middleware.
        chroma_manager: The ChromaDB manager for conversation storage.

    Example:
        >>> agent = BaselineVDBAgent("/path/to/db")
        >>> stats = agent.get_chromadb_stats()
        >>> results = agent.search_past_conversations("hello")
        >>> await agent.clear_agent_memory()  # Clear all stored data
    """

    """A baseline memory agent with vector database integration.

    This agent extends the baseline functionality by integrating with ChromaDB
    for persistent conversation storage and retrieval. It uses middleware for
    both augmentation (storing conversations) and retrieval (searching past
    conversations) to provide long-term memory capabilities.

    Attributes:
        agent: The underlying LangChain agent instance with VDB middleware.
        chroma_manager: The ChromaDB manager for conversation storage.

    Example:
        >>> agent = BaselineVDBAgent("/path/to/db")
        >>> stats = agent.get_chromadb_stats()
        >>> results = agent.search_past_conversations("hello")
        >>> await agent.clear_agent_memory()  # Clear all stored data
    """

    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR) -> None:
        """Initialize the baseline VDB agent.

        Creates a ChromaDB manager for persistent storage and sets up a LangChain
        agent with vector database middleware for conversation augmentation
        and retrieval.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
                Defaults to BASELINE_CHROMADB_DIR from config.
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
        """Return ChromaDB statistics.

        Retrieves current statistics from the ChromaDB conversation collection,
        providing insights into the amount of stored conversation data.

        Returns:
            Dictionary containing database statistics:
                - total_conversation_turns: Number of conversation turns stored.

        Returns:
            Empty dictionary if chroma_manager is not initialized.
        """
        if not self.chroma_manager:
            return {}

        return {
            "total_conversation_turns": self.chroma_manager.conversation_collection.count()
        }

    def search_past_conversations(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search past conversations manually.

        Performs a semantic search through stored conversations using the
        provided query to find relevant past interactions.

        Args:
            query: Search query for finding relevant conversations.
            n_results: Maximum number of results to return. Defaults to 5.

        Returns:
            List of conversation dictionaries matching the search query.
            Each dictionary contains conversation metadata and content.

        Returns:
            Empty list if chroma_manager is not initialized.
        """
        if not self.chroma_manager:
            return []

        return self.chroma_manager.search_conversations(query, n_results)

    async def clear_agent_memory(self):
        """Clear all stored conversations from ChromaDB.

        Removes all conversation data from the ChromaDB collection,
        effectively resetting the agent's persistent memory state.
        This operation is irreversible and will delete all stored
        conversation history.
        """
        self.chroma_manager.clear_collection()
