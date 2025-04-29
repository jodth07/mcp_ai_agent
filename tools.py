from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Gets the weather in a city."""
    return f"The weather in {city} is sunny and 85\u00b0F."

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

all_tools = [search_tool, wiki_tool, get_weather]