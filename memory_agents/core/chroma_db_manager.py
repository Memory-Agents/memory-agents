from typing import Any, Dict, List
from chromadb import Client
from chromadb.config import Settings
from datetime import datetime


class ChromaDBManager:
    """Manages ChromaDB integration for conversational RAG"""

    def __init__(self, persist_directory: str):
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
        # Single collection for all conversations
        self.conversation_collection = self.client.get_or_create_collection(
            name="baseline_conversations", metadata={"hnsw:space": "cosine"}
        )

    def add_conversation_turn(
        self, user_message: str, assistant_message: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Adds a complete conversation turn (user + assistant) to ChromaDB"""
        self.message_counter += 1
        timestamp = datetime.now().isoformat()

        if metadata is None:
            metadata = {}

        # Combine both messages for complete context
        conversation_text = f"User: {user_message}\n\nAssistant: {assistant_message}"

        metadata.update(
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
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
        """Searches in conversation history"""
        total_count = self.conversation_collection.count()
        if total_count == 0:
            return []

        results = self.conversation_collection.query(
            query_texts=[query], n_results=min(n_results, total_count)
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Formats ChromaDB search results"""
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
        self.client.delete_collection("baseline_conversations")
        self._get_or_create_conversation_collection()
