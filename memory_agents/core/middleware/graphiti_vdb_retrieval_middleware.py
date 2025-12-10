from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime

from memory_agents.core.chroma_db_manager import ChromaDBManager
from memory_agents.core.middleware.graphiti_retrieval_middleware_utils import (
    GraphitiRetrievalMiddlewareUtils,
)
from langchain_community.document_compressors.flashrank_rerank import (
    FlashrankRerank,
    Ranker,
)

from memory_agents.core.middleware.vdb_retrieval_middlware_utils import (
    VDBRetrievalMiddlewareUtils,
)


class GraphitiVDBRetrievalMiddleware(
    AgentMiddleware, GraphitiRetrievalMiddlewareUtils, VDBRetrievalMiddlewareUtils
):
    def __init__(
        self, graphiti_tools: dict[str, BaseTool], chroma_manager: ChromaDBManager
    ):
        super().__init__()
        self.chroma_manager: ChromaDBManager = chroma_manager
        ranker = Ranker()
        self.reranker: FlashrankRerank = FlashrankRerank(client=ranker, top_n=5)
        self.graphiti_tools = graphiti_tools

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        nodes, memory_facts = self._retrieve_graphiti_with_user_message(state)
        retrieval_context_graphiti = self._build_graphiti_augmentation_context_message(
            (nodes, memory_facts)
        )
        documents = self._retrieve_chroma_db_with_user_message(state)
        retrieval_context_vdb = (
            self._build_vdb_augmentation_context_message(documents)
            if documents
            else None
        )
        retrieval_context = (
            retrieval_context_graphiti + "\n\n" + retrieval_context_vdb
            if retrieval_context_vdb
            else retrieval_context_graphiti
        )
        if retrieval_context:
            system_message = SystemMessage(content=retrieval_context)
            state["messages"].append(system_message)
        return None
