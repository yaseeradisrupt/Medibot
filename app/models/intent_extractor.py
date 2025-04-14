"""Intent extraction model for medical queries."""
import json
from openai import AsyncOpenAI
from config.config import OPEN_AI_MODEL, INTENT_EXTRACTOR_SYSTEM_PROMPT

# Initialize OpenAI client
openai_client = AsyncOpenAI()

async def extract_query_intent(question, context=None):
    """
    Extract query intent and relevant entities from a medicine-related question.
    
    Args:
        question (str): The user's question
        context (list, optional): Previous conversation messages for context
        
    Returns:
        dict: Dictionary with extracted entities and query intent
    """
    system_prompt = INTENT_EXTRACTOR_SYSTEM_PROMPT
    
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
        print("============Query intent extraction response:===================", response)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting query intent: {e}")
        return {}