import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

from core.run_agent import run_agent
from core.agents.baseline import agent

logger = logging.getLogger()


async def main():
    logger.info("Hello from memory-agents!")
    logger.info(
        "In order to run agents import and execute the according `run()` function."
    )
    logger.info(
        "Or run the test cases in the `tests/` folder with `pytest .` in the root directory."
    )
    logger.info("This is a test execution of the baseline agent:")

    response = await run_agent(agent, "Hello World!", "1")
    logger.info("Response from baseline agent:", response)

    response = await run_agent(agent, "What have i said in my first message?", "1")
    logger.info("Response from baseline agent:", response)


if __name__ == "__main__":
    asyncio.run(main())
