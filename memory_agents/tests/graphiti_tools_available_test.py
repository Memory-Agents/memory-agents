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
