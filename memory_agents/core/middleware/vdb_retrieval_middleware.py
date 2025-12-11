from typing import Any
from memory_agents.core.chroma_db_manager import ChromaDBManager
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage

from memory_agents.core.middleware.vdb_retrieval_middlware_utils import (
    VDBRetrievalMiddlewareUtils,
)

from langchain_community.document_compressors.flashrank_rerank import (
    FlashrankRerank,
    Ranker,
)
from langgraph.runtime import Runtime


class VDBRetrievalMiddleware(AgentMiddleware, VDBRetrievalMiddlewareUtils):
    """Middleware for retrieving relevant memories from vector database before model processing.

    This middleware searches ChromaDB for relevant past conversations based on
    the user's latest message and injects this context as a system message to
    inform the AI's response. It includes reranking to prioritize the most relevant results.

    Attributes:
        chroma_manager (ChromaDBManager): Manager for ChromaDB operations.
        reranker (FlashrankRerank): Reranking component for improving result relevance.
    """

    def __init__(self, chroma_manager: ChromaDBManager):
        """Initialize the VDB retrieval middleware.

        Args:
            chroma_manager (ChromaDBManager): Manager for ChromaDB vector storage operations.
        """
        super().__init__()
        self.chroma_manager: ChromaDBManager = chroma_manager
        ranker = Ranker()
        self.reranker: FlashrankRerank = FlashrankRerank(client=ranker, top_n=5)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Retrieve relevant memories from VDB and inject them as context before model processing.

        This method searches ChromaDB for relevant past conversations based on
        the user's latest message, builds a context message, and injects it as
        a system message to inform the AI's response.

        Args:
            state (AgentState): The current agent state containing messages.
            runtime (Runtime): The LangChain runtime instance.

        Returns:
            dict[str, Any] | None: Always returns None as state is modified in place.
        """
        documents = self._retrieve_chroma_db_with_user_message(state)
        retrieval_context = (
            self._build_vdb_augmentation_context_message(documents)
            if documents
            else None
        )
        if retrieval_context:
            system_message = SystemMessage(content=retrieval_context)
            state["messages"].append(system_message)
        return None
