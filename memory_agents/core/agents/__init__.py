# -*- coding: utf-8 -*-
"""Memory agents core module.

This module provides various agent implementations for memory management and retrieval
in conversational AI systems. It includes baseline agents, vector database enhanced agents,
and Graphiti knowledge graph integration agents.

The module contains the following agent types:
- BaselineAgent: Basic memory agent without persistent storage
- BaselineVDBAgent: Agent with ChromaDB vector database integration
- GraphitiAgent: Agent with Graphiti knowledge graph integration
- GraphitiChromaDBAgent: Hybrid agent combining both Graphiti and ChromaDB

Example:
    Basic usage of a baseline agent:

    >>> from memory_agents.core.agents import BaselineAgent
    >>> agent = BaselineAgent()
    >>> # Agent is ready for use

    Advanced usage with vector database:

    >>> from memory_agents.core.agents import BaselineVDBAgent
    >>> agent = BaselineVDBAgent()
    >>> stats = agent.get_chromadb_stats()
    >>> print(f"Database stats: {stats}")

"""
