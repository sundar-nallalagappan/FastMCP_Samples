from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams, StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
from dotenv import load_dotenv
load_dotenv()

import logging
import uuid
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    filename="adk_mcp_app.log",   # 👈 THIS creates a file
    filemode="a"
)

logger = logging.getLogger("adk.mcp.app")

def new_trace_context():
    return {
        "trace_id": str(uuid.uuid4()),
        "start_time": time.time(),
    }    

math_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="python",
            args=["mathserver.py"]
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
    model="gemini-2.5-flash",
    tools=[
        math_toolset,
        weather_toolset,
    ],
    instruction="""
    You can:
    - Solve math using the math MCP server
    - Get weather information using the weather MCP server

    Use tools when appropriate.
    """
)

async def main():
    trace = new_trace_context()
    trace_id = trace["trace_id"]
    
    logger.info(
        "agent_request_started",
        extra={
            "trace_id": trace_id,
            "agent": agent.name,
            "model": agent.model,
        }
    )
    try:
        # Setup session service
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="mcp_app",
            user_id="user1",
            session_id="session1"
        )
        
        # Create runner
        runner = Runner(
            agent=agent,
            app_name="mcp_app",
            session_service=session_service
        )
        
        # Prepare user message
        content = types.Content(parts=[types.Part(text="What is (3+5)*10 and also check the weather of California?")])
        
        # Run agent
        async for event in runner.run_async(
            user_id="user1",
            session_id="session1",
            new_message=content
        ):
            # Log every event type at DEBUG
            logger.debug(
                "agent_event",
                extra={
                    "trace_id": trace_id,
                    "event_type": event.__class__.__name__,
                }
            )
            
            # Tool invocation visibility
            if getattr(event, "tool_call", None) is not None:
                logger.info(
                    "tool_invocation",
                    extra={
                        "trace_id": trace_id,
                        "tool": event.tool_name,
                        "arguments": event.tool_arguments,
                    }
                )
                
            # 🔹 Tool result
            if getattr(event, "tool_result", None) is not None:
                logger.info(
                    "tool_result",
                    extra={
                        "trace_id": trace_id,
                        "tool": event.tool_result.name,
                        "status": "success",
                    }
                )


            if event.is_final_response():
                elapsed = time.time() - trace["start_time"]

                final_text = event.content.parts[0].text

                logger.info(
                    "agent_request_completed",
                    extra={
                        "trace_id": trace_id,
                        "duration_sec": round(elapsed, 3),
                    }
                )
                
                print("Final response:", final_text )
                print(event.content.parts[0].text)
                
                # Token usage (if exposed by model / runtime)
                usage = getattr(event, "usage", None)
                if usage:
                    logger.info(
                        "token_usage",
                        extra={
                            "trace_id": trace_id,
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                        }
                    )
    except Exception as e:
        logger.exception(
            "agent_request_failed",
            extra={"trace_id": trace_id}
        )
        raise
                        
    finally:
        # Explicitly close toolsets to avoid async cleanup issues
        try:
            await math_toolset.close()
            logger.info(
                    "toolset_closed",
                    extra={"trace_id": trace_id, "toolset": 'math_toolset'}
                )
        except Exception:
            logger.warning(
                    "toolset_close_failed",
                    extra={"trace_id": trace_id, "toolset": 'math_toolset'}
                )
        try:
            await weather_toolset.close()
            logger.info(
                    "toolset_closed",
                    extra={"trace_id": trace_id, "toolset": 'weather_toolset'}
                )
        except Exception:
            logger.warning(
                    "toolset_close_failed",
                    extra={"trace_id": trace_id, "toolset": 'weather_toolset'}
                )

asyncio.run(main())
