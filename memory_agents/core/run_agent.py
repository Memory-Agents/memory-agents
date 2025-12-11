"""Agent execution module for running LangChain agents with memory tracking.

This module provides utilities for running OpenAI LangChain agents with
message history and memory tracking capabilities. It integrates with
Langfuse for observability and tracing.

Attributes:
    langfuse: The Langfuse client for observability and tracing.
    langfuse_handler: The Langfuse callback handler for LangChain integration.
    logger: The module logger for logging events and errors.

"""

import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.messages.utils import AnyMessage

from memory_agents.core.utils.message_conversion_utils import (
    ensure_message_content_is_str,
)

load_dotenv()

from langfuse import get_client
from langfuse.langchain import CallbackHandler

try:
    logger = logging.getLogger()

    # Initialize Langfuse client
    langfuse = get_client()

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
    """Run an OpenAI LangChain agent with a full messages history.

    This function executes a LangChain agent with a complete conversation
    history and integrates with Langfuse for observability tracing.

    Args:
        agent: The LangChain agent instance to execute.
        messages: Full chat history in OpenAI-style format:
                  [{"role": "system"|"user"|"assistant", "content": "..."}]
        thread_id: The thread ID for memory tracking and conversation continuity.

    Returns:
        str: The string response from the agent.

    Raises:
        ValueError: If the agent response is invalid or missing messages.
    """
    input_data = {"messages": messages}
    response = await agent.ainvoke(
        input_data,
        {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else [],
        },
    )
    return extract_response_content(response)


async def run_agent(agent, message: str, thread_id: str) -> str:
    """Run an OpenAI LangChain agent with a single user message.

    The thread_id is passed as an argument and directly forwarded to
    configurable.thread_id, so each message uses independent memory.

    Args:
        agent: The LangChain agent instance to execute.
        message: The user message to send to the agent.
        thread_id: The thread ID for memory tracking and conversation continuity.

    Returns:
        str: The string response from the agent.

    Raises:
        ValueError: If the agent response is invalid or missing messages.
    """
    input_data = {"messages": [{"role": "user", "content": message}]}
    response = await agent.ainvoke(
        input_data,
        {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler] if langfuse_handler else [],
        },
    )
    return extract_response_content(response)


def extract_response_content(response: dict[str, List[AnyMessage]]) -> str:
    """Extract the content string from an agent response.

    Args:
        response: The agent response dictionary containing messages.

    Returns:
        str: The extracted content string from the last message.

    Raises:
        ValueError: If no messages are found in the agent response.
    """
    if "messages" in response and len(response["messages"]) > 0:
        response_content = response["messages"][-1].content
        return ensure_message_content_is_str(response_content)
    else:
        raise ValueError("No messages found in agent response")
