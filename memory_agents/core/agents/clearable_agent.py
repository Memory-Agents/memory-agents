from abc import ABC, abstractmethod


class ClearableAgent(ABC):
    @abstractmethod
    def clear_agent_memory(self): ...
