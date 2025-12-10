from typing import Sequence
from langchain.agents.middleware import AgentState
from langchain_community.document_compressors import FlashrankRerank

from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.utils.agent_state_utils import (
    MessageType,
    get_latest_message_from_agent_state,
)
from langchain_core.documents import Document


class VDBRetrievalMiddlewareUtils:
    chroma_manager: ChromaDBManager
    reranker: FlashrankRerank

    def _retrieve_chroma_db_with_user_message(
        self, state: AgentState
    ) -> Sequence[Document]:
        human_message_type = MessageType.HUMAN
        message = get_latest_message_from_agent_state(state, human_message_type)

        chroma_query = message.content

        similar_conversations = self.chroma_manager.search_conversations(
            chroma_query, n_results=20
        )

        if not similar_conversations:
            raise ValueError("No documents could be retrieved from VDB")

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
    ) -> str:
        augmentation_context = ""

        if not reranked_docs:
            raise ValueError("No documents returned from reranker")

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
