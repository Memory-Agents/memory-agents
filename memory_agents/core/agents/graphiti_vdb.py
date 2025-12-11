# -*- coding: utf-8 -*-
"""Hybrid memory agent with Graphiti knowledge graph and vector database.

This module provides a memory agent implementation that combines Graphiti
knowledge graph with ChromaDB vector database for comprehensive memory
capabilities. The agent uses both structured knowledge graph storage and
semantic vector search for optimal memory retrieval.

Example:
    Creating and using a hybrid agent:

    >>> from memory_agents.core.agents.graphiti_vdb import GraphitiVDBAgent
    >>> agent = await GraphitiVDBAgent.create("/path/to/db")
    >>> # Agent is ready for use with hybrid memory
    >>> stats = agent.get_chromadb_stats()
    >>> results = agent.search_past_conversations("hello")

"""

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
    """A hybrid memory agent with Graphiti knowledge graph and vector database.

    This agent combines the structured knowledge representation of Graphiti
    with the semantic search capabilities of ChromaDB to provide comprehensive
    memory functionality. It uses multiple middleware components for optimal
    information storage and retrieval.

    Attributes:
        agent: The underlying LangChain agent instance with hybrid middleware.
        chroma_manager: The ChromaDB manager for conversation storage.

    Example:
        >>> agent = await GraphitiVDBAgent.create("/path/to/db")
        >>> stats = agent.get_chromadb_stats()
        >>> results = agent.search_past_conversations("hello")
        >>> await agent.clear_agent_memory()  # Clear all stored data
    """

    def __init__(self):
        """Initialize the hybrid agent.

        Creates a new instance with agent and chroma_manager attributes
        set to None. The actual initialization happens in the async create()
        method to allow for async operations during setup.
        """
        self.agent: Any = None
        self.chroma_manager: ChromaDBManager = None

    @classmethod
    async def create(cls, persist_directory: str = GRAPHITI_VDB_CHROMADB_DIR) -> Self:
        """Create and initialize a hybrid agent instance.

        This class method handles the async initialization process, including
        setting up ChromaDB, retrieving Graphiti MCP tools, and configuring
        the LangChain agent with multiple middleware components for hybrid
        memory operations.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
                Defaults to GRAPHITI_VDB_CHROMADB_DIR from config.

        Returns:
            An initialized GraphitiVDBAgent instance ready for use.

        Note:
            This method must be called instead of __init__ to properly
            initialize the agent with async operations.
        """
        self = cls()

        self.chroma_manager = ChromaDBManager(persist_directory)

        graphiti_tools_all = await self._get_graphiti_mcp_tools(is_read_only=False)
        self.agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_MEMORY_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                GraphitiVDBRetrievalMiddleware(graphiti_tools_all, self.chroma_manager),
                GraphitiAugmentationMiddleware(graphiti_tools_all),
                VDBAugmentationMiddleware(self.chroma_manager),
            ],
        )
        return self

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
        provided query to find relevant past interactions from the vector database.

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
        """Clear all stored data from both ChromaDB and Graphiti.

        Removes all conversation data from the ChromaDB collection and all
        entities, relationships, and episodes from the knowledge graph,
        effectively resetting the agent's complete memory state. This operation
        is irreversible and will delete all stored memory.

        Note:
            This method clears both the vector database (via chroma_manager)
            and the knowledge graph (via inherited clear_graph() method).
        """
        self.chroma_manager.clear_collection()
        await self.clear_graph()
