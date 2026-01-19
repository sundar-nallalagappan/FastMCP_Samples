import asyncio
import logging
import time
import uuid
import sys
import json
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StreamableHTTPConnectionParams,
    StdioServerParameters,
)
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

from google.genai import types

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool

# =========================================================
# ENV
# =========================================================
load_dotenv()

# =========================================================
# LOGGING (STDOUT & JSON FORMATTING)
# =========================================================

# Custom JSON Formatter for EKS log collectors (Fluentbit/Datadog/CloudWatch)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add 'extra' fields if they exist
        if hasattr(record, "extra_info"):
            log_record.update(record.extra_info)
        return json.dumps(log_record)

logger = logging.getLogger("adk.mcp.app")
logger.setLevel(logging.INFO)

# Route to Stdout
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


# =========================================================
# OBSERVABILITY PLUGIN (MEMORY-LEAK FIXED)
# =========================================================

class ObservabilityPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="observability")
        # FIXED: Removed self._tool_start_times = {} 
        # class-level dicts cause memory leaks in long-running pods.

    # ------------------ AGENT ------------------

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ):
        trace_id = str(uuid.uuid4())
        callback_context.state["trace_id"] = trace_id
        callback_context.state["agent_start"] = time.time()

        logger.info(
            "agent_started",
            extra={"extra_info": {
                "trace_id": trace_id,
                "agent": agent.name,
                "model": agent.model,
            }},
        )

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ):
        start_time = callback_context.state.get("agent_start")
        duration = time.time() - start_time if start_time else 0

        logger.info(
            "agent_completed",
            extra={"extra_info": {
                "trace_id": callback_context.state.get("trace_id"),
                "duration_sec": round(duration, 3),
            }},
        )

    # ------------------ MODEL ------------------

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ):
        callback_context.state["llm_start"] = time.time()

        logger.info(
            "llm_call_started",
            extra={"extra_info": {
                "trace_id": callback_context.state.get("trace_id"),
                "model": llm_request.model,
            }},
        )

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response
    ):
        start_time = callback_context.state.get("llm_start")
        duration = time.time() - start_time if start_time else 0

        usage = getattr(llm_response, "usage", None)
        payload = {
            "trace_id": callback_context.state.get("trace_id"),
            "duration_sec": round(duration, 3),
        }

        if usage:
            payload.update({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            })

        logger.info("llm_call_completed", extra={"extra_info": payload})

    # ------------------ TOOLS (FIXED) ------------------
    async def before_tool_callback(self, **kwargs):
        # The ADK sometimes passes the context as 'callback_context' or 'tool_context'
        ctx: CallbackContext = kwargs.get("callback_context") or kwargs.get("tool_context")
        tool: BaseTool = kwargs.get("tool")
        
        if not ctx:
            # If for some reason ADK doesn't pass context, log without trace_id to avoid crash
            logger.warning("tool_call_started_no_context", extra={"extra_info": {"tool": getattr(tool, "name", "unknown")}})
            return

        tool_name = getattr(tool, "name", "unknown")
        state_key = f"tool_start_{tool_name}"
        ctx.state[state_key] = time.time()

        logger.info(
            "tool_call_started",
            extra={"extra_info": {
                "trace_id": ctx.state.get("trace_id"),
                "tool": tool_name,
                "arguments": kwargs.get("tool_args") or kwargs.get("tool_input")
            }},
        )
        
    async def after_tool_callback(self, **kwargs):
        ctx: CallbackContext = kwargs.get("callback_context") or kwargs.get("tool_context")
        tool: BaseTool = kwargs.get("tool")
        
        if not ctx:
            return

        tool_name = getattr(tool, "name", "unknown")
        state_key = f"tool_start_{tool_name}"
        start_time = ctx.state.get(state_key)
        duration = round(time.time() - start_time, 3) if start_time else None

        logger.info(
            "tool_call_completed",
            extra={"extra_info": {
                "trace_id": ctx.state.get("trace_id"),
                "tool": tool_name,
                "duration_sec": duration,
                "output": kwargs.get("tool_output")
            }},
        )   

    async def on_error_callback(
        self, *, callback_context: CallbackContext, error: Exception
    ):
        logger.error(
            "agent_error",
            extra={"extra_info": {
                "trace_id": callback_context.state.get("trace_id"),
                "error": str(error),
            }},
            exc_info=True
        )


# =========================================================
# MCP TOOLSETS & AGENT SETUP (SAME AS BEFORE)
# =========================================================

math_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["mathserver.py"],
        )
    )
)

weather_toolset = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:8000/mcp"
    )
)

agent = Agent(
    name="adk_mcp_agent",
    model="gemini-2.5-flash", # Updated to a current valid model string
    tools=[math_toolset, weather_toolset],
    instruction="Solve math and check weather using tools.",
)

# =========================================================
# MAIN
# =========================================================

async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="mcp_app",
        user_id="user1",
        session_id="session1",
    )

    runner = Runner(
        agent=agent,
        app_name="mcp_app",
        session_service=session_service,
        plugins=[ObservabilityPlugin()],
    )

    content = types.Content(
        parts=[types.Part(text="What is (3+5)*10 and check the weather of California?")]
    )

    final_text = ""
    async for event in runner.run_async(
        user_id="user1",
        session_id="session1",
        new_message=content,
    ):
        if getattr(event, "content", None):
            final_text = event.content.parts[0].text

    print(f"\nFinal response: {final_text}")

    await math_toolset.close()
    await weather_toolset.close()

if __name__ == "__main__":
    asyncio.run(main())