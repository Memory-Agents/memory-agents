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
import chromadb
from chromadb.config import Settings
from datetime import datetime

from memory_agents.core.config import BASELINE_CHROMADB_DIR, BASELINE_MODEL_NAME


BASELINE_CHROMADB_SYSTEM_PROMPT = """You are a memory agent that helps the user to solve tasks.
Your conversation history is automatically stored and retrieved to provide context.

When relevant past conversations are found, they will be included in your context to help you:
- Remember previous discussions and user preferences
- Maintain continuity across conversations
- Provide more personalized and contextual responses

You do not need to write memory, only read from the according tool, in order ot 
Focus on helping the user effectively by using the provided context when relevant."""


class ChromaDBManager:
    """Manages ChromaDB integration for conversational RAG"""

    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR):
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=persist_directory,
            )
        )

        # Single collection for all conversations
        self.conversation_collection = self.client.get_or_create_collection(
            name="baseline_conversations", metadata={"hnsw:space": "cosine"}
        )

        self.message_counter = self.conversation_collection.count()

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


class RAGEnhancedAgentMiddleware(AgentMiddleware):
    """Middleware that enriches context with ChromaDB results BEFORE response"""

    def __init__(self, chroma_manager: ChromaDBManager):
        super().__init__()
        self.chroma_manager = chroma_manager
        self.reranker = FlashrankRerank(top_n=5)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # State is a dict with 'messages' key
        messages = state.get("messages", [])
        if not messages:
            return None

        # Get the latest user message
        user_message = state["messages"][-1] if state["messages"][-1] else None

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
                relevance_score = doc.metadata.get("relevance_score", 0)
                if relevance_score > 0.3:  # Relevance threshold
                    timestamp = doc.metadata.get("timestamp", "unknown")
                    rag_context += f"\n[Conversation {i}] (relevance: {relevance_score:.2f}, date: {timestamp}):\n"
                    rag_context += f"{doc.page_content}\n"

        # Inject context if relevant
        if rag_context:
            # Inject context by adding a system message to the state
            context_message = SystemMessage(content=rag_context)
            state["messages"].append(context_message)
            return None  # State is modified in-place

        return None


class ChromaDBStorageMiddleware(AgentMiddleware):
    """Middleware that stores complete conversation in ChromaDB AFTER response"""

    def __init__(self, chroma_manager: ChromaDBManager):
        super().__init__()
        self.chroma_manager = chroma_manager
        self.pending_user_message = None

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Capture user message before model processes it
        messages = state.get("messages", [])
        if not messages:
            return None

        self.pending_user_message =  state["messages"][-1] if state["messages"][-1] else None

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Stores complete conversation turn after model response"""

        if not self.pending_user_message:
            return None

        # Get the latest assistant message
        messages = state.get("messages", [])
        
        assistant_message =  state["messages"][-1] if state["messages"][-1] else None

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


class BaselineAgent:
    """Baseline agent with ChromaDB RAG integration"""

    def __init__(self, persist_directory: str = BASELINE_CHROMADB_DIR) -> None:
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
        """Returns ChromaDB statistics"""
        if not self.chroma_manager:
            return {}

        return {
            "total_conversation_turns": self.chroma_manager.conversation_collection.count()
        }

    def search_past_conversations(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Allows manual search in past conversations"""
        if not self.chroma_manager:
            return []

        return self.chroma_manager.search_conversations(query, n_results)


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
