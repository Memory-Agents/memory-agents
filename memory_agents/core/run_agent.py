import logging
from dotenv import dotenv_values, load_dotenv

load_dotenv()

from langfuse import get_client
from langfuse.langchain import CallbackHandler

try:
    # Initialize Langfuse client
    langfuse = get_client()

    logger = logging.getLogger()

    # Verify connection
    if langfuse.auth_check():
        logger.info("Langfuse client is authenticated and ready!")
    else:
        logger.error("Langfuse client authentication failed.")
        raise Exception("Langfuse client authentication failed.")

    langfuse_handler = CallbackHandler()
except:
    logger.error(
        "Could not initialize Langfuse, please check if Langfuse is running on correctly."
    )

    langfuse_handler = None
    # Initialize Langfuse CallbackHandler for Langchain (tracing)


async def run_agent_messages(
    agent,
    messages: list[dict[str, str]],
    thread_id: str,
) -> str:
    """
    Run an OpenAI LangChain agent with a full messages history.
    Build configuration with Langfuse callback handler if available.

    Args:
        agent: The LangChain agent instance.
        messages: Full chat history in OpenAI-style format:
                  [{"role": "system"|"user"|"assistant", "content": "..."}]
        thread_id: The thread ID for memory tracking.

    Returns:
        The string response from the agent.
    """
    input_data = {"messages": messages}
    response = agent.invoke(
        input_data,
        {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else [],
        },
    )
    return extract_response_content(response)


# The thread_id is passed as an argument and directly forwarded to configurable.thread_id, so each message uses independent memory.
async def run_agent(agent, message: str, thread_id: str) -> str:
    """
    Run an OpenAI LangChain agent with a single user message and return the string response.

    Args:
        agent: The LangChain agent instance.
        message: The user message to send to the agent.
        thread_id: The thread ID for memory tracking.
    Returns:
        The string response from the agent.
    """
    input_data = {"messages": [{"role": "user", "content": message}]}
    response = agent.invoke(
        input_data,
        {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else [],
        },
    )
    return extract_response_content(response)


def extract_response_content(response: dict[str, str]) -> str:
    if "messages" in response and len(response["messages"]) > 0:
        return response["messages"][-1].content
    else:
        raise ValueError("No messages found in agent response")
