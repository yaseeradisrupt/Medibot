from logging import config
from dotenv import dotenv_values
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
from typing import List
from tavily import TavilyClient
from openai.types.chat import ChatCompletionMessageParam  # import the correct type
from typing import cast
import json
from google.cloud import bigquery

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[str]
    datasource:str
    query_info: Optional[dict]

config = dotenv_values(".env")
#Step1: Setup Open AI

collection = get_milvus_collection()
llm = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=config.get("TAVILY_API_KEY", "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"))
client_bigquery = bigquery.Client.from_service_account_json("key.json")

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



def generate_system_prompt(context) -> dict:
    """Generates the system message with structured formatting."""
    return {
        "role": "system",
        "content": f"""
        You are a medical assistant with the knowledge and tone of a highly experienced doctor. 
        The following information represents your internal medical knowledge:

        {context}

        Your responses should be **structured, professional, concise, and informative.**  
        If the user's query is unclear, **ask clarifying questions** to ensure accurate understanding.  

        1. For simple greetings, casual questions, or clarification requests:
           - Respond naturally and conversationally without special formatting
           - Keep responses concise and friendly
           - Example: "Hello! How can I assist you with medical information today?"

        2. For detailed medical queries about conditions, symptoms, treatments, etc.:
            - Use **Markdown with custom tags** for structured responses.
            - **Do NOT merge words** (e.g., `symptomsofdiabetes` → `symptoms of diabetes`).
            - **Use single spaces** between words.
            - **Include relevant sections** based on the query (all sections are optional):
            
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
        
        If you don't know the answer to a medical question, respond with:
        `"I don't have enough information to answer that question accurately. You may want to consult a doctor for additional help."`

        For non-medical questions, politely respond: `"I specialize in medical topics. I cannot provide information on that."`

        **Important Restriction:**  
        - Questions about medication pricing, availability, pharmaceutical information, and drug-related topics ARE considered medical topics and should be answered if information is available.
        - For follow-up questions, consider the context of the entire conversation. A short follow-up like "what about in India?" or "price in India?" is asking about the previously mentioned medication.
        - Only respond with "I specialize in medical topics. I cannot provide information on that." if the question is clearly unrelated to health, medicine, or pharmaceuticals (e.g., sports, politics, entertainment).
        """
    }

async def route_question(state):
    chat_history = build_chat_history(state["messages"])
    question = state["messages"][-1].content
    system_prompt = f"""
        You are an expert router that determines whether a user's question should be answered using:
        1. A medical vector database (for medical knowledge, conditions, symptoms, treatments)
        2. A BigQuery database (for ANY medicine pricing, availability, or comparison queries)
        3. Web search (for recent medicine news, approvals, research)
        4. Not at all (for non-medical topics)

        Your decision rules are as follows:

        1. If the question is a **general greeting** or **non-medical**, respond with:
        {{ "datasource": "unknown", "message": "I specialize in medical topics. How can I assist you with medical information today?" }}

        2. If the question is **not related to medical topics**, respond with:
        {{ "datasource": "unknown", "message": "I specialize in medical topics. I cannot provide information on that." }}

        3. Route to **"bigquery"** if the question is about:
        - Medication pricing (in any country or region)
        - Comparing prices
        - Finding cheapest/most expensive medicines
        - Listing medications for a treatment with price information
        - Any question that needs the structured medicine database
        - Follow-up questions about prices or medications

        4. Route to **"web_search"** if the question is about:
        - Recent medicine approvals or recalls
        - Clinical trials or research news
        - Regulatory updates
        - New treatments or breakthrough information
        - Medicine market information or industry news
        - Historical information about diseases, pandemics, or medical events

        5. Route to **"vectorstore"** if the question is about:
        - General medical knowledge
        - Symptoms, conditions, or diseases
        - Treatment approaches (not specific to pricing)
        - Medical advice or information
        - Side effects or interactions

        **Important Rules:**
        - Only return a raw JSON object with exactly two keys: "datasource" and "message".
        - For any medicine pricing or comparison question, always route to "bigquery"
        - For follow-up questions, consider the entire conversation context
        - Never include explanations, extra text, or formatting.

        Conversation history:
        {chat_history}

        User's latest message:
        {question}

        Based on the conversation above, determine the appropriate data source
        """

    formatted_messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if "role" not in msg:
                print("Fixing missing role:", msg)
                msg["role"] = "user"  # Default to user if missing
            formatted_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})

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
def web_search(state):
    # query = build_chat_history(state["messages"])
    # if len(query) > 400:
    #     query = query[-400:]
    if("context" not in state):
        state["context"] = []
    message= state["messages"][-1]    
    if isinstance(message, dict):
        role = message.get("role", "user")
        content = message.get("content", "")
    elif hasattr(message, "type"):  # probably a LangChain Message
        role = "user" if message.type == "human" else "assistant"
        content = message.content
    else:
        # Fallback, treat as user message
        role = "user"
        content = str(message)    
    state["context"].append( tavily_client.get_search_context(query=content))
    print("tavily search response",state["context"][-1])
    return state

async def rag_context(state):
    messages = state["messages"]
    context = get_context(messages[-1].content)
    if("context" not in state):
        state["context"] = []
    state["context"].append( context)
    return state

async def extract_query_intent(question, context=None):
    """
    Extract query intent and relevant entities from a medicine-related question.
    
    Args:
        question (str): The user's question
        context (list, optional): Previous conversation messages for context
        
    Returns:
        dict: Dictionary with extracted entities and query intent
    """
    system_prompt = """
    Extract all relevant entities and the query intent from the given question about medicine pricing.
    Use the conversation context if available to resolve references and ambiguities.
    
    The database 'chatbot' has a table called 'medicines_db' with the following schema:
    - medicine_name (text): Name of the medicine
    - medicine_price (numeric): Price of the medicine
    - country (text): Country where the medicine is sold
    - treatment (text): What condition the medicine treats
    
    If the question refers to a follow-up like "show me more", "next", "give me 10 more", use the context to infer:
    - the same medicine_name as before
    - offset = previous limit + previous offset
    - update the limit if the user mentions a different numbe

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
    print("=====Full prompt for intent extraction======:", full_prompt)
    try:
        response = await openai_client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[{"role": "system", "content": full_prompt}],
            response_format={"type": "json_object"}
        )
        print("============Query intent extraction response:===================", response)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting query intent: {e}")
        return {}

async def generate_dynamic_sql(question, extracted_intent, table_schema=None):
    """
    Generate SQL query dynamically based on the question and extracted intent.
    
    Args:
        question (str): The original question
        extracted_intent (dict): Extracted entities and intent
        table_schema (dict, optional): Schema information
        
    Returns:
        str: Generated SQL query
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
    1. Return ONLY the SQL query — do NOT include any explanation, comments, or markdown formatting like ```sql.
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
        response =await openai_client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}]
        )
        print("============SQL generation response:===================", response)
        sql_query = response.choices[0].message.content.strip()
        sql_query =sql_query.strip("`sql\n").strip("`")
        # Basic validation that we received SQL
        if not sql_query.upper().startswith("SELECT"):
            print(f"Warning: Generated SQL does not start with SELECT: {sql_query}")
            # Simple fallback query
            return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"
        
        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"

async def bigquery_lookup(state):

    print("====================bigquery_lookup state================", state)
    """
    Process medicine pricing query using BigQuery.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        dict: Updated state with BigQuery results
    """
    if "context" not in state:
        state["context"] = []
    
    # Extract current question
    message = state["messages"][-1]
    if isinstance(message, dict):
        question = message.get("content", "")
    elif hasattr(message, "content"):
        question = message.content
    else:
        question = str(message)
    
    # Extract query intent
    extracted_intent = await extract_query_intent(question, state["messages"])
    print("==========extracted_intent======:", extracted_intent)
    
    # Generate SQL dynamically
    sql_query = await generate_dynamic_sql(question, extracted_intent)
    print("==========Generated SQL query======:", sql_query)
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
        
        # Generate title
        if len(formatted_results) > 0:
            result_type = "results"
        else:
            result_type = "No results found"
        
        # Add to context
        context_text = f"""
            ### Medicine Database Query Results: {result_type}
            
            {json.dumps(formatted_results, indent=2) if formatted_results else "No data found matching your query."}
            
            Query Details:
            ```sql
            {sql_query}
            ```
        """
        
        state["context"].append(context_text)
        state["query_info"] = {
            "question": question,
            "extracted_intent": extracted_intent,
            "sql_query": sql_query,
            "results": formatted_results
        }
    except Exception as e:
        error_message = f"Error executing BigQuery: {str(e)}"
        state["context"].append(error_message)
        state["query_info"] = {
            "error": error_message,
            "sql_query": sql_query,
            "extracted_intent": extracted_intent
        }
    
    return state

async def stream_unknown(state, config, writer):
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
        # yield yield_data
    # return {"messages": state["messages"]}

async def generate_response(state, config, writer):
    print("=====generate_response initial state====", state)
    messages = state["messages"]
    context=state["context"][-1]
        
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
    print("System message before OpenAI call:", system_message)
    full_messages = [system_message] + formatted_messages
    full_response = ""
    print("Final messages before OpenAI call:", full_messages)

    async for msg_chunk in stream_tokens(full_messages):
        full_response += msg_chunk["content"]
        metadata = {**config["metadata"], "tags": ["response"]}
        writer((msg_chunk, metadata))

    return {"messages": messages + [{"role": "assistant", "content": full_response}]}

def build_chat_history(messages: list):
    history = ""
    for msg in messages[-5:]:  # last 5 messages
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        elif hasattr(msg, "type"):  # probably a LangChain Message
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
        else:
            # Fallback, treat as user message
            role = "user"
            content = str(msg)

        history += f"{role.capitalize()}: {content}\n"

    return history



builder = StateGraph(state_schema=GraphState)
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
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
