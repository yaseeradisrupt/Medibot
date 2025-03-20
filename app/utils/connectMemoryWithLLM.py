from load_dotenv import load_dotenv
from openai import OpenAI
from typing_extensions import Annotated, TypedDict
from typing import Optional
from langgraph.graph import StateGraph, START, END
from app.config.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPEN_AI_MODEL
from app.utils.milvusCollection import get_milvus_collection
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: Optional[str]

load_dotenv()
#Step1: Setup Open AI

collection = get_milvus_collection()
llm = OpenAI(api_key=OPENAI_API_KEY)
def get_embedding(text):
    """Generates an embedding for the given text using OpenAI."""
    response = llm.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def search_milvus(query_embedding, top_k=3):
    """Searches Milvus for the most similar documents."""
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 16},  # Adjust nprobe as needed
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["content"],
    )
    return [hit.entity.get("content") for hit in results[0]]
def get_context(query):
    query_embedding = get_embedding(query)
    retrieved_contexts = search_milvus(query_embedding)
    return "\n".join(retrieved_contexts)

async def stream_tokens(messages):
    print("🔹 Received messages:", messages)
    response = await openai_client.chat.completions.create(
        messages=messages, model=OPEN_AI_MODEL, stream=True
    )

    role = None
    buffer = "" 
    async for chunk in response:
        delta = chunk.choices[0].delta
        if delta.role is not None:
            role = delta.role
        if delta.content:
            buffer += delta.content
            words = buffer.split(" ")
            if len(words) > 1:  # If there's a space, we have at least one complete word
                for word in words[:-1]:  # Yield all complete words
                    yield {"role": role, "content": word + " "}
                buffer = words[-1]  # Keep the last (possibly incomplete) word in the buffer
            else:
                continue 
    if buffer:
        yield {"role": role, "content": buffer}        

async def call_model(state, config, writer):
    messages = state["messages"]
    context = get_context(messages[-1].content)
    
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if "role" not in msg:
                print("Fixing missing role:", msg)
                msg["role"] = "user"  # Default to user if missing
            formatted_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
    
    system_message = generate_system_prompt(context)
    full_messages = [system_message] + formatted_messages
    full_response = ""
    print("Final messages before OpenAI call:", full_messages)

    async for msg_chunk in stream_tokens(full_messages):
        full_response += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["response"]}
        writer((msg_chunk, metadata))

    return {"messages": messages + [{"role": "assistant", "content": full_response}]}

def generate_system_prompt(context: str) -> dict:
    """Generates the system message with structured formatting."""
    return {
        "role": "system",
        "content": f"""
        You are a medical assistant with the knowledge and tone of a highly experienced doctor. 
        The following information represents your internal medical knowledge:

        {context}

        Your responses should be **structured, professional, concise, and informative.**  
        If the user's query is unclear, **ask clarifying questions** to ensure accurate understanding.  

        **Handling Greetings:**  
        - If the user says **"hi," "hello," "hey," or similar casual greetings**, respond politely and ask how you can help.  
        - Example:  
          - User: "Hi"  
          - Response: "Hello! How can I assist you today?"

        **Guidelines for Responding:**
        - Use **Markdown with custom tags** for structured responses.
        - **Do NOT merge words** (e.g., `symptomsofdiabetes` → `symptoms of diabetes`).
        - **Use single spaces** between words.
        - **Ensure proper section formatting** for easy parsing.

        **Response Format:**
        
        **TITLE** <Main Topic> \n

        **DESCRIPTION** <Brief overview of the condition> \n

        **Symptoms** \n
        - <Symptom 1> \n
        - <Symptom 2> \n
        - <Symptom 3> \n

        **Causes** \n
        - **Cause 1** \n
        - **Cause 2** \n

        **Treatment**\n
        - <Treatment 1>\n
        - <Treatment 2>\n
        
        If you **do not know the answer**, respond with:  
        `"I don't know the answer. You may want to consult a doctor for additional help."`
        """
    }

builder = StateGraph(state_schema=GraphState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)