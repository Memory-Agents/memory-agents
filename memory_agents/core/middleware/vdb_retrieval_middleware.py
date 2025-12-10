from typing import Any
from memory_agents.core.chroma_db_manager import ChromaDBManager
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage

from memory_agents.core.middleware.vdb_retrieval_middlware_utils import (
    VDBRetrievalMiddlewareUtils,
)

from langchain_community.document_compressors import FlashrankRerank
from langgraph.runtime import Runtime


class VDBRetrievalMiddleware(AgentMiddleware, VDBRetrievalMiddlewareUtils):
    def __init__(self, chroma_manager: ChromaDBManager):
        super().__init__()
        self.chroma_manager: ChromaDBManager = chroma_manager
        self.reranker: FlashrankRerank = FlashrankRerank(top_n=5)

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        documents = self._retrieve_chroma_db_with_user_message(state)
        retrieval_context = self._build_augmentation_context_message(documents)

        system_message = SystemMessage(content=retrieval_context)
        state["messages"].append(system_message)
        return None
