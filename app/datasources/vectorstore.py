"""Vector database operations."""
from utils.message_formatting import get_last_message_content
from openai import OpenAI
from config.config import OPENAI_API_KEY, EMBEDDING_MODEL
from app.utils.milvusCollection import get_milvus_collection

llm = OpenAI(api_key=OPENAI_API_KEY)
collection = get_milvus_collection()

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
# This would be your actual retrieval function implementation
def get_context(query):
    """
    Retrieve context from vector database based on query.
    
    Args:
        query (str): Query string
        
    Returns:
        str: Retrieved context
    """
    query_embedding = get_embedding(query)
    retrieved_contexts = search_milvus(query_embedding)
    return "\n".join(retrieved_contexts)


async def rag_context(state):
    """
    Retrieve context from vector database.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        dict: Updated state with context
    """
    if "context" not in state:
        state["context"] = []
    
    content = get_last_message_content(state["messages"])
    context = get_context(content)
    
    state["context"].append(context)
    return state
