from typing import Any
from langchain.agents.graph import CompiledStateGraph
from langchain.agents.schema import AgentState

OpenAIAgent = CompiledStateGraph[AgentState[str], Any, Any, Any]