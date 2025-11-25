async def run_agent(agent, message: str) -> dict[str, str]:
    """
    Run an OpenAI LangChain agent with a single user message and return the string response.
    """
    input_data = {"messages": [{"role": "user", "content": message}]}

    # Invoke the agent with the input data and a fixed thread ID for memory tracking
    response = agent.invoke(input_data, {"configurable": {"thread_id": "1"}})

    return extract_response_content(response)


def extract_response_content(response: dict[str, str]) -> str:
    if "messages" in response and len(response["messages"]) > 0:
        return response["messages"][-1].content
    else:
        raise ValueError("No messages found in agent response")
