"""Web search using Tavily API."""
from tavily import TavilyClient
from config.config import TAVILY_API_KEY
from utils.message_formatting import get_last_message_content

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def web_search(state):
    """
    Perform a web search using Tavily API.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        dict: Updated state with web search context
    """
    if "context" not in state:
        state["context"] = []
        
    content = get_last_message_content(state["messages"])
    
    # Perform web search
    search_results = tavily_client.get_search_context(query=content)
    state["context"].append(search_results)
    
    print("Tavily search response", state["context"][-1])
    return state
