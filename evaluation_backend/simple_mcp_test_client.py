import asyncio
import json
import os
from typing import Optional, List, Any

import httpx
from pydantic import BaseModel, Field
from openai import AsyncOpenAI


# ==== Pydantic model for tool invocation ====
class MCPToolCall(BaseModel):
    tool: str = Field(..., description="Name of the MCP tool to call")
    args: dict = Field(..., description="Arguments for the MCP tool")


# ==== MCP Client (SSE-based) ====
class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def call_tool(self, tool: str, args: dict) -> Any:
        """Send a tool request to the MCP server and get a response."""
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {"tool": tool, "args": args}
            resp = await client.post(f"{self.base_url}/invoke", json=payload)
            resp.raise_for_status()
            return resp.json()


# ==== Chat client with GPT-4o ====
class ChatClient:
    def __init__(self, mcp: MCPClient, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.mcp = mcp
        self.history: List[dict] = [
            {"role": "system", "content": "You are an assistant connected to an MCP server. "
                                          "If a user request requires tool use, respond with a JSON object "
                                          "matching the MCPToolCall schema instead of natural language."}
        ]

    async def chat_once(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.history,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            tool_call = MCPToolCall(**data)
            print(f"[MCP] → Calling tool: {tool_call.tool}({tool_call.args})")
            result = await self.mcp.call_tool(tool_call.tool, tool_call.args)
            print(f"[MCP] ← Result: {result}")
            self.history.append({"role": "assistant", "content": json.dumps(result)})
        except Exception:
            # If not structured as tool call, just print it
            print(f"Assistant: {content}")
            self.history.append({"role": "assistant", "content": content})


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY")

    # Point this to your MCP server’s HTTP endpoint
    mcp = MCPClient("http://localhost:8000")
    chat = ChatClient(mcp, api_key)

    print("Connected. Type your message (Ctrl+C to quit).")
    while True:
        try:
            user_input = input("\nYou: ")
            await chat.chat_once(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
