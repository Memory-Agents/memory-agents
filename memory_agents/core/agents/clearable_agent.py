# -*- coding: utf-8 -*-
"""Abstract base class for clearable memory agents.

This module defines the interface for agents that can clear their memory
state. All memory agent implementations should inherit from this base class
to ensure consistent memory management capabilities.

Example:
    Implementing a clearable agent:

    >>> from memory_agents.core.agents.clearable_agent import ClearableAgent
    >>> class MyAgent(ClearableAgent):
    ...     async def clear_agent_memory(self):
    ...         # Clear memory implementation
    ...         pass

"""

from abc import ABC, abstractmethod


class ClearableAgent(ABC):
    """Abstract base class for agents with clearable memory.

    This interface ensures that all memory agent implementations provide
    a consistent method for clearing their stored memory state, whether
    it's in-memory checkpointing, vector database storage, or knowledge
    graph data.

    Example:
        >>> class MyAgent(ClearableAgent):
        ...     async def clear_agent_memory(self):
        ...         # Clear all stored memory
        ...         await self.storage.clear()
    """

    @abstractmethod
    async def clear_agent_memory(self):
        """Clear the agent's stored memory.

        This abstract method must be implemented by all concrete agent classes
        to provide memory clearing functionality. The specific implementation
        depends on the type of storage used (in-memory, vector database, etc.).

        Raises:
            NotImplementedError: If not implemented by concrete class.
        """
        ...
