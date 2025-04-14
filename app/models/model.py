from logging import config
from dotenv import dotenv_values
from openai import OpenAI
from typing_extensions import Annotated, TypedDict
from typing import Optional, List, Dict, Any, Union
from langgraph.graph import StateGraph, START, END
from app.config.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPEN_AI_MODEL
from app.utils.milvusCollection import get_milvus_collection
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage
from openai import AsyncOpenAI
from tavily import TavilyClient
from openai.types.chat import ChatCompletionMessageParam
import json
from google.cloud import bigquery

# Initialize clients
config = dotenv_values(".env")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
llm = OpenAI(api_key=OPENAI_API_KEY)
collection = get_milvus_collection()
tavily_client = TavilyClient(api_key=config.get("TAVILY_API_KEY"))
client_bigquery = bigquery.Client.from_service_account_json("key.json")

# Define the state schema
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[str]
    tool_calls: Optional[List[Dict]]
    function_results: Optional[Dict[str, Any]]
    query_info: Optional[Dict]

# Tool functions
def get_embedding(text):
    """Generates an embedding for the given text using OpenAI."""
    response = llm.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def search_milvus(query_embedding, top_k=3):
    """Searches Milvus for the most similar documents."""
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 16},
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
    """Get context from vector database based on query."""
    query_embedding = get_embedding(query)
    retrieved_contexts = search_milvus(query_embedding)
    return "\n".join(retrieved_contexts)

def get_web_search_results(query):
    """Get search results from Tavily."""
    results = tavily_client.get_search_context(query=query)
    return results

async def build_chat_history(messages: list):
    """Build a string representation of chat history."""
    history = ""
    for msg in messages[-5:]:  # last 5 messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        elif hasattr(msg, "type"):
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
        else:
            role = "user"
            content = str(msg)
        history += f"{role.capitalize()}: {content}\n"
    return history

async def stream_tokens(messages):
    """Stream tokens from the OpenAI API."""
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
            if len(words) > 1:
                for word in words[:-1]:
                    yield {"role": role, "content": word + " "}
                buffer = words[-1]
            else:
                continue 
    if buffer:
        yield {"role": role, "content": buffer}

def generate_system_prompt(context="") -> dict:
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
        - Include relevant sections based on the query (all sections are optional):

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

        **Important Restriction:**  
        - Questions about medication pricing, availability, pharmaceutical information, and drug-related topics ARE considered medical topics and should be answered if information is available.
        - For follow-up questions, consider the context of the entire conversation. A short follow-up like "what about in India?" or "price in India?" is asking about the previously mentioned medication.
        - Only respond with "I specialize in medical topics. I cannot provide information on that." if the question is clearly unrelated to health, medicine, or pharmaceuticals (e.g., sports, politics, entertainment).
        """
    }

# Define tool specifications for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_vector_database",
            "description": "Query the medical knowledge vector database for information about medical conditions, symptoms, treatments, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The medical query to look up in the knowledge database"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent medical news, approvals, research, or other up-to-date medical information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The medical query to search for on the web"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_medicine_database",
            "description": "Query the medicine database for pricing, availability, or comparison of medications",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The medicine pricing or availability query"
                    },
                    "medicine_name": {
                        "type": "string",
                        "description": "The name of the medicine to look up (if specified)"
                    },
                    "country": {
                        "type": "string",
                        "description": "The country to get pricing information for (if specified)"
                    },
                    "treatment": {
                        "type": "string",
                        "description": "The medical condition or treatment to find medicines for (if specified)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# LangGraph nodes
async def route_with_tools(state):
    """
    Use OpenAI function calling to route the query to the appropriate tool
    with improved context management
    """
    # Reset context for each new query to prevent bleeding of previous information
    state["context"] = []
    
    messages = state["messages"]
    
    # Format messages for OpenAI
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if "role" not in msg:
                msg["role"] = "user"
            formatted_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
    
    # Add system message
    system_message = {
        "role": "system",
        "content": """You are a medical assistant that helps route user queries to the appropriate tool.
        - Use the query_vector_database tool for general medical knowledge questions about symptoms, conditions, diseases, and treatment approaches.
        - Use the search_web tool for questions about recent medical news, approvals, research, or regulatory updates.
        - Use the query_medicine_database tool for ANY question related to medication pricing, availability, comparing prices, or finding medicines for specific treatments with price information.
        - For non-medical questions, do not use any tools and respond directly with "I specialize in medical topics. I cannot provide information on that."
        - For greetings, respond directly without using tools.
        
        """
    }
    
    full_messages = [system_message] + formatted_messages
    
    # Call OpenAI with function calling
    response = await openai_client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=full_messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    # Check if the model wants to call functions
    if message.tool_calls:
        state["tool_calls"] = message.tool_calls
        return state
    else:
        # Model decided to respond directly
        state["messages"].append({"role": "assistant", "content": message.content})
        return state

async def call_tools(state):
    """
    Execute the tool calls requested by the LLM with improved context management
    """
    function_results = {}
    
    # Clear existing context to prevent context bleed
    state["context"] = []
    
    for tool_call in state["tool_calls"]:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name == "query_vector_database":
            query = arguments.get("query")
            result = get_context(query)
            function_results[tool_call.id] = result
            state["context"].append(result)
            
        elif function_name == "search_web":
            query = arguments.get("query")
            result = get_web_search_results(query)
            function_results[tool_call.id] = result
            state["context"].append(result)
            
        elif function_name == "query_medicine_database":
            query = arguments.get("query")
            medicine_name = arguments.get("medicine_name", "")
            country = arguments.get("country", "")
            treatment = arguments.get("treatment", "")
            
            # Extract query intent
            extracted_intent = await extract_query_intent(query, state["messages"][-1:])  # Only use the latest message
            
            # Enhance with additional parameters if provided
            if medicine_name:
                extracted_intent["medicine_name"] = medicine_name
            if country:
                extracted_intent["country"] = country
            if treatment:
                extracted_intent["treatment"] = treatment
                
            # Generate SQL dynamically
            sql_query = await generate_dynamic_sql(query, extracted_intent)
            
            try:
                # Execute query
                query_job = client_bigquery.query(sql_query)
                results = query_job.result()
                
                # Format results
                formatted_results = []
                for row in results:
                    result_item = {}
                    for field in row.keys():
                        if hasattr(row[field], 'strftime') and callable(getattr(row[field], 'strftime')):
                            result_item[field] = row[field].strftime("%Y-%m-%d")
                        else:
                            result_item[field] = row[field]
                    formatted_results.append(result_item)
                
                # Generate context text - specific to this query only
                context_text = f"""
                    ### Medicine Database Query Results for: {query}
                    
                    {json.dumps(formatted_results, indent=2) if formatted_results else "No data found matching your query."}
                """
                
                state["context"].append(context_text)
                function_results[tool_call.id] = json.dumps(formatted_results)
                state["query_info"] = {
                    "question": query,
                    "extracted_intent": extracted_intent,
                    "sql_query": sql_query,
                    "results": formatted_results
                }
            except Exception as e:
                error_message = f"Error executing BigQuery: {str(e)}"
                state["context"].append(error_message)
                function_results[tool_call.id] = error_message
    
    state["function_results"] = function_results
    return state

async def generate_final_response(state, config, writer):
    """
    Generate the final response using the tool results with improved context management
    """
    messages = state["messages"]
    
    # Format original messages
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if "role" not in msg:
                msg["role"] = "user"
            formatted_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
    
    # Add the assistant's message with tool calls - only for the current interaction
    if "tool_calls" in state:
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": state["tool_calls"]
        }
        formatted_messages.append(assistant_msg)
        
        # Add tool responses - only for the current interaction
        for tool_call in state["tool_calls"]:
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": state["function_results"].get(tool_call.id, "No result")
            }
            formatted_messages.append(tool_msg)
    
    # Get ONLY the most recent context from the state
    # Important: Only use the context generated for the current question
    current_context = state.get("context", [])
    latest_context = current_context[-1] if current_context else ""
    
    # Prepare the system message with only the relevant context
    system_message = generate_system_prompt(latest_context)
    full_messages = [system_message] + formatted_messages
    
    # Add a specific instruction to only respond to the current question
    full_messages.append({
        "role": "system",
        "content": "Important: Only respond to the user's most recent question. Do not include information from previous questions unless directly relevant to the current query."
    })
    
    # Generate and stream the response
    full_response = ""
    async for msg_chunk in stream_tokens(full_messages):
        full_response += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["response"]}
        writer((msg_chunk, metadata))
    
    return {"messages": messages + [{"role": "assistant", "content": full_response}]}
# Helper functions for BigQuery
async def extract_query_intent(question, context=None):
    """
    Extract query intent and relevant entities from a medicine-related question.
    """
    system_prompt = """
    Extract all relevant entities and the query intent from the given question about medicine pricing.
    Use the conversation context if available to resolve references and ambiguities.
    
    The database 'chatbot' has a table called 'medicines_db' with the following schema:
    - medicine_name (text): Name of the medicine
    - medicine_price (numeric): Price of the medicine
    - country (text): Country where the medicine is sold
    - treatment (text): What condition the medicine treats
    
    Return a JSON object with all identified entities and the query intent.
    Include any information that would be needed to construct a SQL query against 
    this table schema.
    
    Return ONLY the JSON object without additional explanation.
    """
    
    context_text = ""
    if context:
        for msg in context[-5:]:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                context_text += f"{content}\n"
            elif hasattr(msg, "content"):
                context_text += f"{msg.content}\n"
    
    full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuery: {question}"
    
    try:
        response = await openai_client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[{"role": "system", "content": full_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting query intent: {e}")
        return {}

async def generate_dynamic_sql(question, extracted_intent, table_schema=None):
    """
    Generate SQL query dynamically based on the question and extracted intent.
    """
    schema_info = table_schema or {
        "table_name": "chatbot.medicines_db",
        "columns": ["medicine_name", "medicine_price", "country", "treatment"]
    }
    
    system_prompt = f"""
    You are an expert SQL generator. Generate a SQL query for BigQuery that addresses the user's question.

    Table Schema:
    Table name: {schema_info["table_name"]}
    Columns: {", ".join(schema_info["columns"])}
    
    User Question: {question}
    
    Extracted Intent and Entities: {json.dumps(extracted_intent)}
    
    Important Guidelines:
    1. Return ONLY the SQL query without explanation or markdown formatting
    2. Use standard SQL syntax compatible with Google BigQuery
    3. When filtering text values, use LOWER() on both sides to make searches case-insensitive
    4. Use LIKE with % wildcards for partial matches on text fields
    5. Include a reasonable LIMIT clause (default to 10 if not specified)
    6. Sort results in a way that best answers the user's question
    7. For ambiguous questions, provide a general query that likely addresses the intent
    8. For follow-up questions, use the extracted context to make appropriate references
    
    Generate the SQL query now:
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}]
        )
        sql_query = response.choices[0].message.content.strip()
        
        # Basic validation
        if not sql_query.upper().startswith("SELECT"):
            return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"
        
        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"

# Decision function to determine the next node
def should_call_tools(state):
    """
    Determine if we need to call tools or go straight to response
    """
    if "tool_calls" in state and state["tool_calls"]:
        return "call_tools"
    else:
        return "end"

# Build LangGraph
builder = StateGraph(state_schema=GraphState)
builder.add_node("route_with_tools", route_with_tools)
builder.add_node("call_tools", call_tools)
builder.add_node("generate_final_response", generate_final_response)

# Add edges
builder.add_edge(START, "route_with_tools")
builder.add_conditional_edges(
    "route_with_tools",
    should_call_tools,
    {
        "call_tools": "call_tools",
        "end": "generate_final_response"
    }
)
builder.add_edge("call_tools", "generate_final_response")
builder.add_edge("generate_final_response", END)

# Initialize memory and compile graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)