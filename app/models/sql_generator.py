"""SQL generation model for BigQuery."""
import json
from openai import AsyncOpenAI
from config.config import OPEN_AI_MODEL, SQL_GENERATOR_SYSTEM_PROMPT, BIGQUERY_TABLE_SCHEMA

# Initialize OpenAI client
openai_client = AsyncOpenAI()

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
    schema_info = table_schema or BIGQUERY_TABLE_SCHEMA
    
    system_prompt = SQL_GENERATOR_SYSTEM_PROMPT.format(
        table_name=schema_info["table_name"],
        columns=", ".join(schema_info["columns"]),
        question=question,
        extracted_intent=json.dumps(extracted_intent)
    )
    
    try:
        response = await openai_client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[{"role": "system", "content": system_prompt}]
        )
        print("============SQL generation response:===================", response)
        sql_query = response.choices[0].message.content.strip()
        
        # Basic validation that we received SQL
        if not sql_query.upper().startswith("SELECT"):
            print(f"Warning: Generated SQL does not start with SELECT: {sql_query}")
            # Simple fallback query
            return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"
        
        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return f"SELECT * FROM {schema_info['table_name']} LIMIT 10"