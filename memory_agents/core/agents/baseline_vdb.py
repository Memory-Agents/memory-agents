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
from langchain.agents.middleware import (
    AgentState,
    AgentMiddleware,
)
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from memory_agents.core.agents.clearable_agent import ClearableAgent
from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.config import BASELINE_CHROMADB_DIR, BASELINE_MODEL_NAME


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


class RAGEnhancedAgentMiddleware(AgentMiddleware):
    """Middleware that enriches context with ChromaDB results BEFORE response.

    This middleware searches ChromaDB for similar past conversations and injects
    relevant context into the agent's state before the model processes the request.
    It uses FlashrankRerank to improve the relevance of retrieved conversations.

    Attributes:
        chroma_manager: The ChromaDB manager for conversation storage and retrieval.
        reranker: The FlashrankRerank for improving search result relevance.

    Example:
        >>> middleware = RAGEnhancedAgentMiddleware(chroma_manager)
        >>> # Middleware will automatically enrich context during agent execution
    """

    def __init__(self, chroma_manager: ChromaDBManager):
        """Initialize the RAG enhancement middleware.

        Args:
            chroma_manager: The ChromaDB manager instance for searching conversations.
        """
        super().__init__()
        self.chroma_manager = chroma_manager
        self.reranker = FlashrankRerank(top_n=5)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Enrich agent context with relevant past conversations.

        Searches ChromaDB for conversations similar to the current user message,
        reranks the results for relevance, and injects the top results as context
        into the agent's state before model processing.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            None if state is modified in-place, otherwise a dictionary of updates.
        """
        # State is a dict with 'messages' key
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get the latest user message
        user_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                user_message = msg
                break

        if not user_message:
            return None

        query = user_message.content

        # Search in ChromaDB
        similar_conversations = self.chroma_manager.search_conversations(
            query, n_results=20
        )

        if not similar_conversations:
            return None

        docs_to_rerank = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in similar_conversations
        ]

        reranked_docs = self.reranker.compress_documents(docs_to_rerank, query)

        # Build enriched context
        rag_context = ""

        if reranked_docs:
            rag_context += "\n--- Similar Past Conversations ---\n"
            for i, doc in enumerate(reranked_docs, 1):
                if i <= 3:  # ranking is more stable than absolute scoring
                    timestamp = doc.metadata.get("timestamp", "unknown")
                    rag_context += f"\n[Conversation {i}], date: {timestamp}):\n"
                    rag_context += f"{doc.page_content}\n"

        # Inject context if relevant
        if rag_context:
            # Inject context by adding a system message to the state
            context_message = SystemMessage(content=rag_context)
            state["messages"].append(context_message)
            return None  # State is modified in-place

        return None


class ChromaDBStorageMiddleware(AgentMiddleware):
    """Middleware that stores complete conversation in ChromaDB AFTER response.

    This middleware captures user messages before model processing and stores
    the complete conversation turn (user message + assistant response) in
    ChromaDB after the model generates its response.

    Attributes:
        chroma_manager: The ChromaDB manager for conversation storage.
        pending_user_message: The user message awaiting storage after response.

    Example:
        >>> middleware = ChromaDBStorageMiddleware(chroma_manager)
        >>> # Middleware will automatically store conversations after each turn
    """

    def __init__(self, chroma_manager: ChromaDBManager):
        """Initialize the ChromaDB storage middleware.

        Args:
            chroma_manager: The ChromaDB manager instance for storing conversations.
        """
        super().__init__()
        self.chroma_manager = chroma_manager
        self.pending_user_message = None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Capture user message before model processing.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            None - this method only captures the user message for later storage.
        """
        # Capture user message before model processes it
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get the latest user message
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                self.pending_user_message = msg.content
                break

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Stores complete conversation turn after model response.

        Stores the captured user message along with the assistant's response
        in ChromaDB as a complete conversation turn with metadata.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            None - this method only stores the conversation in ChromaDB.
        """
        if not self.pending_user_message:
            return None

        # Get the latest assistant message
        messages = state.get("messages", [])
        assistant_message = None
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                assistant_message = msg
                break

        if assistant_message:
            # Store complete conversation in ChromaDB
            self.chroma_manager.add_conversation_turn(
                user_message=self.pending_user_message,
                assistant_message=assistant_message.content,
                metadata={
                    "thread_id": str(state.get("thread_id", "")),
                },
            )

            # Reset pending message
            self.pending_user_message = None

        return None


class BaselineVDBAgent(ClearableAgent):
    """Baseline agent with ChromaDB RAG integration.

    This agent combines ChromaDB vector database storage with RAG (Retrieval-Augmented
    Generation) capabilities. It automatically stores conversations and retrieves
    relevant past context during conversations.

    Attributes:
        agent: The underlying LangChain agent with middleware.
        chroma_manager: The ChromaDB manager for conversation storage and retrieval.

    Example:
        >>> agent = BaselineVDBAgent()
        >>> stats = agent.get_chromadb_stats()
        >>> print(f"Database stats: {stats}")
        >>> results = agent.search_past_conversations("python", n_results=3)
    """

    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR) -> None:
        """Initialize the baseline VDB agent.

        Args:
            persist_directory: Directory path for ChromaDB persistence.
        """
        self.chroma_manager = ChromaDBManager(persist_directory)

        agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_CHROMADB_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                RAGEnhancedAgentMiddleware(self.chroma_manager),
                ChromaDBStorageMiddleware(self.chroma_manager),
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


"""

# Usage example
def main():
    # Create baseline agent with ChromaDB
    agent = BaselineAgent()

    # Display stats
    stats = agent.get_chromadb_stats()
    print(f"ChromaDB Stats: {stats}")

    # Example manual search (optional)
    results = agent.search_past_conversations("python programming", n_results=3)
    print(f"\nFound {len(results)} similar conversations")
    for result in results:
        print(f"- {result['metadata'].get('timestamp')}: {result['content'][:100]}...")

    # Agent is now ready to use with RAG
    return agent


if __name__ == "__main__":
    main()"""
