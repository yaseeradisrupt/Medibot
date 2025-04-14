"""BigQuery database operations."""
import json
from google.cloud import bigquery
from utils.message_formatting import get_last_message_content
from models.intent_extractor import extract_query_intent
from models.sql_generator import generate_dynamic_sql

# Initialize BigQuery client
client_bigquery = bigquery.Client('key.json')

async def bigquery_lookup(state):
    """
    Process medicine pricing query using BigQuery.
    
    Args:
        state (dict): The current graph state
        
    Returns:
        dict: Updated state with BigQuery results
    """
    print("====================bigquery_lookup state================", state)
    
    if "context" not in state:
        state["context"] = []
    
    # Extract current question
    question = get_last_message_content(state["messages"])
    
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