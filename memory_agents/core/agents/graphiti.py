from typing import Self
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from memory_agents.core.agents.interfaces.clearable_agent import ClearableAgent
from memory_agents.core.agents.graphiti_base_agent import GraphitiBaseAgent
from memory_agents.config import BASELINE_MODEL_NAME

from memory_agents.core.config import BASELINE_MEMORY_PROMPT
from memory_agents.core.middleware.graphiti_augmentation_middleware import (
    GraphitiAugmentationMiddleware,
)
from memory_agents.core.middleware.graphiti_retrieval_middleware import (
    GraphitiRetrievalMiddleware,
)

class GraphitiAgent(GraphitiBaseAgent, ClearableAgent):
    def __init__(self):
        """Initialize the Graphiti agent."""
        self.agent = None

    @classmethod
    async def create(cls) -> Self:
        self = cls()
        graphiti_tools_all = await self._get_graphiti_mcp_tools(is_read_only=False)
        self.agent = create_agent(
            model=BASELINE_MODEL_NAME,
            system_prompt=BASELINE_MEMORY_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                GraphitiAugmentationMiddleware(graphiti_tools_all),
                GraphitiRetrievalMiddleware(graphiti_tools_all),
            ],
        )
        return self

    async def clear_agent_memory(self):
        """Clear all data from the Graphiti knowledge graph."""
        await self.clear_graph()
