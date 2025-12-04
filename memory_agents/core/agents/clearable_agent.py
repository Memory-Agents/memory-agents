from abc import ABC, abstractmethod


class ClearableAgent(ABC):
    @abstractmethod
    async def clear_agent_memory(self): ...
