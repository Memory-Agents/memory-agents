from memory_agents.core.run_agent import run_agent
from memory_agents.core.agents.baseline import agent 
import asyncio

def main():
    print("Hello from memory-agents!")
    print("In order to run agents import and execute the according `run()` function.")
    print("This is a test execution of the baseline agent:")
    
    response = asyncio.run(run_agent(agent, "Hello World!"))
    print("Response from baseline agent:", response)


if __name__ == "__main__":
    main()
