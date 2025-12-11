"""Configuration module for memory agents system.

This module contains configuration constants and prompts for the memory
agents system, including model settings, database paths, and system prompts.

Attributes:
    GRAPHITI_MCP_URL (str): URL for the Graphiti MCP server endpoint.
    BASELINE_MODEL_NAME (str): Default model name for baseline operations.
    GRAPHITI_VDB_CHROMADB_DIR (str): Directory path for Graphiti vector database.
    BASELINE_CHROMADB_DIR (str): Directory path for baseline ChromaDB storage.
    BASELINE_MEMORY_PROMPT (str): System prompt for memory agent operations.
"""

GRAPHITI_MCP_URL = "http://localhost:8000/mcp"
"""str: URL for the Graphiti MCP (Model Context Protocol) server endpoint."""

BASELINE_MODEL_NAME = "gpt-4o-mini"
"""str: Default model name used for baseline operations and comparisons."""

GRAPHITI_VDB_CHROMADB_DIR = "./graphiti_vdb_chroma_memory_db"
"""str: Directory path for storing Graphiti vector database ChromaDB files."""

BASELINE_CHROMADB_DIR = "./baseline_chroma_memory_db"
"""str: Directory path for storing baseline ChromaDB conversation history."""

BASELINE_MEMORY_PROMPT = """You are a memory agent that helps the user to solve tasks.
Your conversation history is automatically stored and retrieved to provide context.

When relevant past conversations are found, they will be included in your context to help you:
- Remember previous discussions and user preferences
- Maintain continuity across conversations
- Provide more personalized and contextual responses

You do not need to manage memory yourself - it is handled automatically.
Focus on helping the user effectively by using the provided context when relevant.

You must follow these steps:
Step 1: Evaluate whether retrieved context is relevant (return yes/no and justification).
Step 2: Produce final answer using only the relevant information.

Return only Step 2 to the user.
"""
"""str: System prompt template for memory agent operations.

This prompt instructs the agent on how to handle conversation memory,
evaluate retrieved context relevance, and structure responses appropriately.
"""
