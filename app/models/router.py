"""Question routing model for the medical bot."""
import json
from openai import AsyncOpenAI
from config.config import OPEN_AI_MODEL, get_roter_system_prompt
from utils.chat_history import build_chat_history
from utils.message_formatting import format_messages

# Initialize OpenAI client
openai_client = AsyncOpenAI()

async def route_question(state):
    """
    Route a question to the appropriate data source.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        dict: Updated state with routing decision
    """
    chat_history = build_chat_history(state["messages"])
    question = state["messages"][-1].content
    system_prompt = get_roter_system_prompt(
        question,
        chat_history
    )   
    
    formatted_messages = format_messages(state["messages"])
    full_messages = [{"role": "system", "content": system_prompt}] + formatted_messages

    print("Full messages before OpenAI call:", full_messages)
    response = await openai_client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=full_messages
    )

    content = response.choices[0].message.content.strip()

    try:
        print("content", content)
        result = json.loads(content)
        if result.get("datasource") == 'unknown':
            state["datasource"] = "unknown"
            state["messages"].append({"role": "assistant", "content": result.get("message")})
        elif result.get("datasource") in ['vectorstore', 'web_search', 'bigquery']:
            state["datasource"] = result.get("datasource")
        print("state in route_question", state)

        return state
    except Exception as e:
        print("Error parsing response:", content)
        raise e

def node_to_decide(state):
    """
    Return the next node to be called based on datasource.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    if state["datasource"] == "web_search":
        return "web_search"
    elif state["datasource"] == "vectorstore":
        return "rag_context"
    elif state["datasource"] == "bigquery":
        return "bigquery"
    else:
        return "unknown"