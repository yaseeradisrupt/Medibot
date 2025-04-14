"""State graph implementation for the medical bot."""
from langchain.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI
from config.config import OPEN_AI_MODEL, MEDICAL_ASSISTANT_SYSTEM_PROMPT
from utils.message_formatting import format_messages
from models.router import route_question, node_to_decide
from datasources.vectorstore import rag_context
from datasources.web_search import web_search
from datasources.bigquery import bigquery_lookup
from typing_extensions import Annotated, TypedDict
from typing import Optional
from langgraph.graph.message import add_messages
from typing import List

# Initialize OpenAI client
openai_client = AsyncOpenAI()
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[str]
    datasource:str
    query_info: Optional[dict]

def generate_system_prompt(context):
    """Generates the system message with structured formatting."""
    return {
        "role": "system",
        "content": MEDICAL_ASSISTANT_SYSTEM_PROMPT.format(context=context)
    }

async def stream_tokens(messages):
    """
    Stream tokens from OpenAI API.
    
    Args:
        messages (list): List of messages to send to API
        
    Yields:
        dict: Token chunks
    """
    stream = await openai_client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=messages,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield {"role": "assistant", "content": chunk.choices[0].delta.content}

async def stream_unknown(state, config, writer):
    """
    Stream unknown response.
    
    Args:
        state (dict): Current state
        config: Configuration
        writer: Response writer
    """
    print("stream_unknown state", state)
    last_msg = state["messages"][-1]
    if isinstance(last_msg, dict):
        last_msg_content = last_msg["content"]
    elif hasattr(last_msg, "content"):
        last_msg_content = last_msg.content
    role = "assistant"
    print("stream_unknown last_msg", last_msg)
    for word in last_msg_content.split(" "):
        yield_data = {"role": role, "content": word + " "}
        metadata = {**config["metadata"], "tags": ["response"]}
        writer((yield_data, metadata))

async def generate_response(state, config, writer):
    """
    Generate response from OpenAI.
    
    Args:
        state (dict): Current state
        config: Configuration
        writer: Response writer
    """
    print("=====generate_response initial state====", state)
    messages = state["messages"]
    context = state["context"][-1]
        
    formatted_messages = format_messages(messages)
    
    system_message = generate_system_prompt(context)
    print("System message before OpenAI call:", system_message)
    full_messages = [system_message] + formatted_messages
    full_response = ""
    print("Final messages before OpenAI call:", full_messages)

    async for msg_chunk in stream_tokens(full_messages):
        full_response += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["response"]}
        writer((msg_chunk, metadata))

    return {"messages": messages + [{"role": "assistant", "content": full_response}]}

def create_graph():
    """
    Create the state graph.
    
    Returns:
        Compiled state graph
    """
    builder = StateGraph(state_schema=GraphState)
    
    # Add nodes
    
    builder.add_node("route_question", route_question)
    builder.add_node("rag_context", rag_context)
    builder.add_node("web_search", web_search)
    builder.add_node("generate_response", generate_response)
    builder.add_node("node_to_decide", node_to_decide)
    builder.add_node("stream_unknown", stream_unknown)
    builder.add_edge(START, "route_question")
    builder.add_node("bigquery_lookup", bigquery_lookup)
    builder.add_conditional_edges("route_question", node_to_decide,{
        "web_search": "web_search",
        "rag_context": "rag_context",
        "bigquery": "bigquery_lookup",
        "unknown":"stream_unknown"
    })
    builder.add_edge("rag_context", "generate_response")
    builder.add_edge("bigquery_lookup", "generate_response")
    builder.add_edge("web_search", "generate_response")
    builder.add_edge("generate_response", END)
    builder.add_edge("stream_unknown", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile graph
    return builder.compile(checkpointer=memory)