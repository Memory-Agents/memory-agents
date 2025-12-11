"""Tests for Graphiti tools availability and filtering.

This module contains tests to verify that the Graphiti MCP (Model Context Protocol)
tools are properly filtered and only the appropriate tools are made available to
agents. The test ensures that dangerous operations like deletion are not exposed
while necessary query and retrieval tools are available.

The test verifies:
- Disallowed tools (delete operations) are not present
- Required tools (search, retrieval) are present
- Optional diagnostic tools are handled appropriately
- No unexpected tools are exposed
"""

import pytest

from memory_agents.core.agents.graphiti_base_agent import GraphitiBaseAgent

ALLOWED_TOOLS = {
    "search_nodes",
    "search_memory_facts",
    "get_episodes",
    "get_entity_edge",
    "get_status",
}

DISALLOWED_TOOLS = {
    # "add_memory", Allow add_memory for now, as sndjg removed it.
    "delete_episode",
    "delete_entity_edge",
    "clear_graph",
}


@pytest.mark.asyncio
async def test_graphiti_tools_available():
    """Test Graphiti MCP tools filtering and availability.

    Verifies that the GraphitiBaseAgent properly filters MCP tools to ensure
    only safe and necessary tools are exposed. This is a security and safety
    test to prevent agents from accessing destructive operations.

    The test checks:
    1. Disallowed tools (delete/clear operations) are not present
    2. Required tools (search/retrieval operations) are present
    3. Optional diagnostic tools are handled appropriately
    4. No unexpected tools are exposed beyond the allowed set

    Args:
        None: Uses module-level constants for allowed/disallowed tools.

    Returns:
        None: Raises AssertionError if tool filtering is incorrect.

    Raises:
        AssertionError: If disallowed tools are present, required tools are missing,
            or unexpected tools are found.
    """
    agent = GraphitiBaseAgent()

    tools = await agent._get_graphiti_mcp_tools()

    tool_names = set(tools.keys())

    # --- Ensure disallowed tools are not present ---
    assert not (DISALLOWED_TOOLS & tool_names), (
        f"Disallowed tools were not filtered out: {DISALLOWED_TOOLS & tool_names}"
    )

    # --- Ensure all allowed tools are present (except optional get_status) ---
    missing_required = (ALLOWED_TOOLS - {"get_status"}) - tool_names
    assert not missing_required, f"Missing required Graphiti tools: {missing_required}"

    # --- Optional diagnostic tool ---
    assert "get_status" in tool_names or True, "get_status tool is optional"

    # --- Optional: ensure no unexpected tools are present ---
    unexpected_tools = tool_names - ALLOWED_TOOLS
    assert not unexpected_tools, f"Unexpected tools found: {unexpected_tools}"
