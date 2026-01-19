from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from dotenv import load_dotenv
load_dotenv()
import os
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

import asyncio
async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
                },
            "weather":{
                "url":"http://localhost:8000/mcp",
                "transport":"streamable-http"
                }
        }
    )
    

    tools = await client.get_tools()  
    print('tools:', tools)  
    model = ChatGroq(model="qwen/qwen3-32b")
    agent=create_agent(model,tools=tools)
    print('agent:', agent)
    math_response=await agent.ainvoke({
        "messages":[{"role":"user","content":"What is (3+5)*10 and also check the weather of california?"}]
    })
    print('math_response-raw', math_response)
    print('\n')
    print('math_response-messages', math_response['messages'])
    print('\n')
    
    ai_messages = [m for m in math_response["messages"] if isinstance(m, AIMessage)]
    print('ai_messages', ai_messages)

    
asyncio.run(main())