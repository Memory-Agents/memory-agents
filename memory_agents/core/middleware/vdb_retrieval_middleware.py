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
    def __init__(self, chroma_manager: ChromaDBManager):
        super().__init__()
        self.chroma_manager: ChromaDBManager = chroma_manager
        ranker = Ranker()
        self.reranker: FlashrankRerank = FlashrankRerank(client=ranker, top_n=5)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
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
