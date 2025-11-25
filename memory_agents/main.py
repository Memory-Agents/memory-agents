import asyncio
from dotenv import load_dotenv

load_dotenv()

from core.run_agent import run_agent
from core.agents.baseline import agent



async def main():
    print("Hello from memory-agents!")
    print("In order to run agents import and execute the according `run()` function.")
    print("This is a test execution of the baseline agent:")

    response = await run_agent(agent, "Hello World!")
    print("Response from baseline agent:", response)

    response = await run_agent(agent, "What have i said in my first message?")
    print("Response from baseline agent:", response)


if __name__ == "__main__":
    asyncio.run(main())
