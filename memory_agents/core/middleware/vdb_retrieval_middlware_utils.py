from typing import Sequence
from langchain.agents.middleware import AgentState
from langchain_community.document_compressors import FlashrankRerank

from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
)
from langchain_core.documents import Document
import logging


class VDBRetrievalMiddlewareUtils:
    """Utility class providing vector database retrieval functionality.

    This class contains helper methods for retrieving relevant conversations
    from ChromaDB based on user queries and building context messages for
    AI responses. It includes reranking to improve result relevance.

    Attributes:
        chroma_manager (ChromaDBManager): Manager for ChromaDB operations.
        reranker (FlashrankRerank): Reranking component for improving result relevance.
        logger: Logger instance for debugging and error reporting.
    """

    chroma_manager: ChromaDBManager
    reranker: FlashrankRerank

    def __init__(self):
        """Initialize the VDB retrieval utilities.

        Sets up the logger instance for debugging and error reporting.
        """
        self.logger = logging.getLogger()

    def _retrieve_chroma_db_with_user_message(
        self, state: AgentState
    ) -> Sequence[Document] | None:
        """Retrieve relevant conversations from ChromaDB based on the user's latest message.

        This method extracts the latest human message, searches ChromaDB for
        similar conversations, and reranks the results to prioritize relevance.

        Args:
            state (AgentState): The current agent state containing messages.

        Returns:
            Sequence[Document] | None: A sequence of reranked Document objects
                containing relevant past conversations, or None if no results found.
        """
        logger = logging.getLogger()

        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)

        if not isinstance(message.content, str):
            chroma_query = str(message.content)
            self.logger.error(
                "The retrieved message content is not a str, this might be unexpected behavior"
            )
        else:
            chroma_query = message.content

        similar_conversations = self.chroma_manager.search_conversations(
            chroma_query, n_results=20
        )

        if not similar_conversations:
            logger.error("No documents could be retrieved from VDB")
            return None

        docs_to_rerank = [
            Document(
                page_content=conversation["content"], metadata=conversation["metadata"]
            )
            for conversation in similar_conversations
        ]

        reranked_docs = self.reranker.compress_documents(docs_to_rerank, chroma_query)
        return reranked_docs

    def _build_vdb_augmentation_context_message(
        self, reranked_docs: Sequence[Document]
    ) -> str | None:
        """Build a context message from reranked conversation documents.

        This method formats the top 3 most relevant past conversations into
        a structured context message that can be injected into the AI's
        conversation to provide relevant background information.

        Args:
            reranked_docs (Sequence[Document]): A sequence of reranked Document
                objects containing relevant past conversations.

        Returns:
            str | None: A formatted context message containing the retrieved
                conversations with timestamps and usage instructions, or None
                if no documents are available.
        """
        augmentation_context = ""

        if not reranked_docs:
            self.logger.error("No documents returned from reranker")
            return None

        augmentation_context += "\n--- Similar Past Conversations ---\n"
        for i, doc in enumerate(reranked_docs, 1):
            if i <= 3:
                timestamp = doc.metadata.get("timestamp", "unknown")
                augmentation_context += f"\n[Conversation {i}], date: {timestamp}):\n"
                augmentation_context += f"{doc.page_content}\n"

        retrieval_context = f"""
            <retrieved_context>
            {augmentation_context}
            </retrieved_context>

            IMPORTANT:
            Only use information from <retrieved_context> if it is clearly relevant to the user's query.
            If it is not relevant, IGNORE it entirely.
            """
        return retrieval_context
