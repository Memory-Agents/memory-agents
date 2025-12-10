from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool


class GraphitiVDBRetrievalMiddleware(AgentMiddleware):
    pass

    def __init__(self, graphiti_tools: dict[str, BaseTool]):
        super().__init__()
        self.graphiti_tools = graphiti_tools
        pass
