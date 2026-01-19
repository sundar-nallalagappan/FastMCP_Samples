from mcp.server.fastmcp import FastMCP

mcp=FastMCP(name="Weather")

@mcp.tool()
async def get_weather(city:str)->str:
    """Get the weather of a city
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string containing the weather information for the specified city
    """
    return f"The weather in {city} is sunny"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")