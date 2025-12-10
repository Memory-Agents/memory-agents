# -*- coding: utf-8 -*-
"""Baseline agent with ChromaDB vector database integration.

This module provides a memory agent implementation that uses ChromaDB for
persistent conversation storage and retrieval. It includes middleware for
RAG (Retrieval-Augmented Generation) functionality and automatic conversation
history management.

Example:
    Creating and using a baseline VDB agent:

    >>> from memory_agents.core.agents.baseline_vdb import BaselineVDBAgent
    >>> agent = BaselineVDBAgent()
    >>> stats = agent.get_chromadb_stats()
    >>> print(f"Database stats: {stats}")

"""

from typing import Any, Dict, List
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent
from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.config import BASELINE_CHROMADB_DIR, BASELINE_MODEL_NAME
from memory_agents.core.middleware.vdb_augmentation_middleware import (
    VDBAugmentationMiddleware,
)
from memory_agents.core.middleware.vdb_retrieval_middleware import (
    VDBRetrievalMiddleware,
)


BASELINE_CHROMADB_SYSTEM_PROMPT = """You are a memory agent that helps the user to solve tasks.
Your conversation history is automatically stored and retrieved to provide context.

When relevant past conversations are found, they will be included in your context to help you:
- Remember previous discussions and user preferences
- Maintain continuity across conversations
- Provide more personalized and contextual responses

You do not need to manage memory yourself - it is handled automatically.
Focus on helping the user effectively by using the provided context when relevant.

You must follow these steps:
Step 1: Evaluate whether retrieved context is relevant (return yes/no and justification).
Step 2: Produce final answer using only the relevant information.

Return only Step 2 to the user.
"""


class BaselineVDBAgent(ClearableAgent):
    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR) -> None:
        """Initialize the baseline VDB agent.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
        """
        self.chroma_manager = ChromaDBManager(persist_directory)

        agent: Any = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_CHROMADB_SYSTEM_PROMPT,
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
