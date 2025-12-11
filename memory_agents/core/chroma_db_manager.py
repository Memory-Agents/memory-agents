"""ChromaDB integration module for conversational RAG systems.

This module provides a manager class for handling ChromaDB operations
including storing, retrieving, and searching conversation history
for retrieval-augmented generation (RAG) applications.

"""

from typing import Any, Dict, List
from chromadb import Client, QueryResult
from chromadb.config import Settings
from datetime import datetime


class ChromaDBManager:
    """Manages ChromaDB integration for conversational RAG.

    This class provides a high-level interface for storing and retrieving
    conversation history using ChromaDB as the vector database backend.
    It handles collection management, document storage, and similarity search.

    Attributes:
        client: The ChromaDB client instance.
        conversation_collection: The ChromaDB collection for storing conversations.
        message_counter: Counter for tracking the number of stored messages.

    Args:
        persist_directory: Directory path for persistent ChromaDB storage.
    """

    def __init__(self, persist_directory: str):
        """Initialize the ChromaDB manager with persistent storage.

        Args:
            persist_directory: Directory path for persistent ChromaDB storage.
        """
        self.client = Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=persist_directory,
            )
        )

        self._get_or_create_conversation_collection()

        self.message_counter = self.conversation_collection.count()

    def _get_or_create_conversation_collection(self):
        """Get or create the conversation collection.

        Creates a single collection for all conversations using cosine
        similarity for HNSW indexing. This method is called during initialization.
        """
        # Single collection for all conversations
        self.conversation_collection = self.client.get_or_create_collection(
            name="baseline_conversations", metadata={"hnsw:space": "cosine"}
        )

    def add_conversation_turn(
        self, user_message: str, ai_message: str, metadata: Dict[str, Any] | None = None
    ) -> None:
        """Add a complete conversation turn to ChromaDB.

        Stores both user and assistant messages as a single document with
        associated metadata for retrieval and context.

        Args:
            user_message: The user's message in the conversation turn.
            ai_message: The assistant's response in the conversation turn.
            metadata: Optional additional metadata to store with the conversation.
        """
        self.message_counter += 1
        timestamp = datetime.now().isoformat()

        if metadata is None:
            metadata = {}

        # Combine both messages for complete context
        conversation_text = f"User: {user_message}\n\nAssistant: {ai_message}"

        metadata.update(
            {
                "user_message": user_message,
                "ai_message": ai_message,
                "timestamp": timestamp,
                "turn_id": self.message_counter,
            }
        )

        self.conversation_collection.add(
            documents=[conversation_text],
            metadatas=[metadata],
            ids=[f"turn_{self.message_counter}"],
        )

    def search_conversations(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search conversation history using semantic similarity.

        Performs a similarity search against stored conversation turns
        to find relevant context for the given query.

        Args:
            query: The search query to find relevant conversations.
            n_results: Maximum number of results to return. Defaults to 5.

        Returns:
            List of dictionaries containing search results with content,
            metadata, distance scores, and document IDs.
        """
        total_count = self.conversation_collection.count()
        if total_count == 0:
            return []

        results = self.conversation_collection.query(
            query_texts=[query], n_results=min(n_results, total_count)
        )

        return self._format_results(results)

    def _format_results(self, results: QueryResult) -> List[Dict[str, Any]]:
        """Format ChromaDB search results into a standardized structure.

        Converts the raw ChromaDB query result into a list of dictionaries
        with consistent structure for easier consumption.

        Args:
            results: The raw QueryResult from ChromaDB.

        Returns:
            List of dictionaries containing formatted search results with
            content, metadata, distance scores, and document IDs.
        """
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append(
                    {
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                        "id": results["ids"][0][i] if results["ids"] else None,
                    }
                )
        return documents

    def clear_collection(self):
        """Clear all conversation data from the collection.

        Deletes the existing conversation collection and creates a new
        empty one, effectively removing all stored conversation history.
        """
        self.client.delete_collection("baseline_conversations")
        self._get_or_create_conversation_collection()
