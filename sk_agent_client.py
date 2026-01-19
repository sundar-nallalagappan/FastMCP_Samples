import asyncio

from semantic_kernel import Kernel
# Note: FunctionCallingStepwisePlanner has been deprecated/removed in recent SK versions
# Use automatic function calling with FunctionChoiceBehavior.Auto() instead
# from semantic_kernel.planners.function_calling_stepwise_planner import FunctionCallingStepwisePlanner
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# MCP imports (native SK MCP support)
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
import os

# -----------------------------
# CONFIG
# -----------------------------

WEATHER_MCP_URL = "http://localhost:8000/mcp"
OPENAI_MODEL = "gpt-4o-mini"  # or gpt-4o / gpt-4.1 / gpt-3.5-turbo

# -----------------------------
# CONFIG
# -----------------------------

WEATHER_MCP_URL = "http://localhost:8000/mcp"

OPENAI_MODEL = "gpt-4o-mini"  # or gpt-4o / gpt-4.1 / gpt-3.5-turbo

# -----------------------------
# MAIN
# -----------------------------

async def main():
    # 1️⃣ Create Semantic Kernel
    kernel = Kernel()

    # 2️⃣ Add OpenAI LLM
    kernel.add_service( 
        OpenAIChatCompletion(
            ai_model_id=OPENAI_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            service_id="my-service-id"
        )
    )

    # 3️⃣ Load MCP server as a plugin (THIS replaces MCPClient)
    plugin = MCPStreamableHttpPlugin(name="Weather",
            description="Weather Plugin",
            url=WEATHER_MCP_URL
        )
    
    # 3️⃣ Load MCP server as a plugin (THIS replaces MCPClient)
    await plugin.connect()

    # 4️⃣ Register plugin with kernel
    kernel.add_plugin(plugin)    

   
        # Create execution settings with automatic tool calling
    execution_settings = PromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )


    # 7️⃣ Invoke agent
    prompt = "What is the weather in California today?"

    result = await kernel.invoke_prompt(
    prompt,
    settings=execution_settings
    )


    print("\n--- FINAL RESPONSE ---")
    print(result)
    
    # 7️⃣ Clean up
    await plugin.close()


if __name__ == "__main__":
    asyncio.run(main())