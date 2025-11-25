from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    system_prompt="You are a memory agent ",
)



