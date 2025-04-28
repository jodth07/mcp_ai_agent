from datetime import datetime

import requests
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, Tool
#
@tool
def query_mcp_weather(city: str) -> str:
    """Query MCP server for weather information by city."""
    try:
        response = requests.post("http://localhost:8000/weather", json={"city": city}, timeout=5)
        if response.ok:
            data = response.json()
            return data.get("forecast", "No forecast data available.")
        else:
            return f"Weather server error: {response.status_code}"
    except Exception as e:
        return f"Failed to connect to weather MCP server: {e}"

# @tool
# def query_mcp_stock(symbol: str) -> str:
#     """Query MCP server for stock price by stock symbol."""
#     try:
#         response = requests.post("http://localhost:8000/stock", json={"symbol": symbol}, timeout=5)
#         if response.ok:
#             data = response.json()
#             return data.get("price_info", "No stock information available.")
#         else:
#             return f"Stock server error: {response.status_code}"
#     except Exception as e:
#         return f"Failed to connect to stock MCP server: {e}"
#
# @tool
# def query_mcp_news(topic: str) -> str:
#     """Query MCP server for latest news headlines on a topic."""
#     try:
#         response = requests.post("http://localhost:8000/news", json={"topic": topic}, timeout=5)
#         if response.ok:
#             data = response.json()
#             return data.get("headlines", "No news available.")
#         else:
#             return f"News server error: {response.status_code}"
#     except Exception as e:
#         return f"Failed to connect to news MCP server: {e}"

@tool
def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Saves structured research data to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"

# @tool
# def search_tool():
#     """Search the web for information."""
#     search = DuckDuckGoSearchRun()
#     return search.run

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)


@tool
def get_weather(city: str) -> str:
    """Gets the weather in a city."""
    return f"The weather in {city} is sunny and 85\u00b0F."

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

all_tools = [save_to_txt, search_tool, wiki_tool, get_weather
             # query_mcp_weather, query_mcp_stock, query_mcp_news,
             ]