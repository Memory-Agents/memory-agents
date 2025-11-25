from memory_agents.utils.openaiagent import OpenAIAgent


async def run_agent(agent: OpenAIAgent, message: str) -> str:
    """
    Run an OpenAI LangChain agent with a single user message and return the string response.
    """
    # Format the message for the agent
    input_data = {
        "messages": [{"role": "user", "content": message}]
    }

    # Invoke the agent asynchronously
    response: str = await agent.invoke(input_data)

    return response
