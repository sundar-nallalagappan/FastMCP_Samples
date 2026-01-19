    import asyncio
    import logging
    import time
    import uuid
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
    # LOGGING (FILE-BASED)
    # =========================================================

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        filename="adk_mcp_app.log",
        filemode="a",
    )

    logger = logging.getLogger("adk.mcp.app")


    # =========================================================
    # OBSERVABILITY PLUGIN (FULL LIFECYCLE TRACING)
    # =========================================================

    class ObservabilityPlugin(BasePlugin):
        def __init__(self):
            super().__init__(name="observability")
            self._tool_start_times = {}  

        # ------------------ AGENT ------------------

        async def before_agent_callback(
            self, *, agent: BaseAgent, callback_context: CallbackContext
        ):
            trace_id = str(uuid.uuid4())
            callback_context.state["trace_id"] = trace_id
            callback_context.state["agent_start"] = time.time()

            logger.info(
                "agent_started",
                extra={
                    "trace_id": trace_id,
                    "agent": agent.name,
                    "model": agent.model,
                },
            )

        async def after_agent_callback(
            self, *, agent: BaseAgent, callback_context: CallbackContext
        ):
            duration = time.time() - callback_context.state["agent_start"]

            logger.info(
                "agent_completed",
                extra={
                    "trace_id": callback_context.state["trace_id"],
                    "duration_sec": round(duration, 3),
                },
            )

        # ------------------ MODEL ------------------

        async def before_model_callback(
            self, *, callback_context: CallbackContext, llm_request: LlmRequest
        ):
            callback_context.state["llm_start"] = time.time()

            logger.info(
                "llm_call_started",
                extra={
                    "trace_id": callback_context.state["trace_id"],
                    "model": llm_request.model,
                },
            )

        async def after_model_callback(
            self, *, callback_context: CallbackContext, llm_response
        ):
            duration = time.time() - callback_context.state["llm_start"]

            usage = getattr(llm_response, "usage", None)

            payload = {
                "trace_id": callback_context.state["trace_id"],
                "duration_sec": round(duration, 3),
            }

            if usage:
                payload.update(
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                )

            logger.info("llm_call_completed", extra=payload)

        # ------------------ TOOLS ------------------
        
        async def before_tool_callback(self, **kwargs):
            callback_context = kwargs.get("callback_context")
            tool = kwargs.get("tool")

            tool_args = (
                kwargs.get("tool_args")
                or kwargs.get("tool_input")
                or kwargs.get("tool_context")
            )

            trace_id = None
            if callback_context is not None:
                trace_id = callback_context.state.get("trace_id")

            tool_name = getattr(tool, "name", "unknown")

            # Store timing OUTSIDE callback_context
            if trace_id is not None:
                self._tool_start_times[(trace_id, tool_name)] = time.time()

            logger.info(
                "tool_call_started",
                extra={
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "arguments": tool_args
                },
            )
            
        async def after_tool_callback(self, **kwargs):
            callback_context = kwargs.get("callback_context")
            tool = kwargs.get("tool")

            trace_id = None
            if callback_context is not None:
                trace_id = callback_context.state.get("trace_id")

            tool_name = getattr(tool, "name", "unknown")

            duration = None
            if trace_id is not None:
                start = self._tool_start_times.pop(
                    (trace_id, tool_name), None
                )
                if start is not None:
                    duration = round(time.time() - start, 3)

            logger.info(
                "tool_call_completed",
                extra={
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "duration_sec": duration
                },
            )   

        # ------------------ ERRORS ------------------

        async def on_error_callback(
            self, *, callback_context: CallbackContext, error: Exception
        ):
            logger.exception(
                "agent_error",
                extra={
                    "trace_id": callback_context.state.get("trace_id"),
                    "error": str(error),
                },
            )


    # =========================================================
    # MCP TOOLSETS
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


    # =========================================================
    # AGENT
    # =========================================================

    agent = Agent(
        name="adk_mcp_agent",
        model="gemini-2.5-flash",
        tools=[math_toolset, weather_toolset],
        instruction="""
    You can:
    - Solve math using the math MCP server
    - Get weather information using the weather MCP server

    Use tools when appropriate.
    """,
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
            parts=[
                types.Part(
                    text="What is (3+5)*10 and also check the weather of California?"
                )
            ]
        )

        final_text = None

        async for event in runner.run_async(
            user_id="user1",
            session_id="session1",
            new_message=content,
        ):
            if getattr(event, "content", None):
                final_text = event.content.parts[0].text

        print("\nFinal response:")
        print(final_text)

        # Cleanup MCP toolsets
        await math_toolset.close()
        await weather_toolset.close()


    if __name__ == "__main__":
        asyncio.run(main())
