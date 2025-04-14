"""Message formatting utilities for the medical bot."""

def format_message(message):
    """Format message to consistent structure.
    
    Args:
        message: The message to format (can be dict, HumanMessage, or other types)
        
    Returns:
        dict: Formatted message with 'role' and 'content' keys
    """
    if isinstance(message, dict):
        if "role" not in message:
            message["role"] = "user"  # Default to user if missing
        return message
    elif hasattr(message, "type"):  # probably a LangChain Message
        role = "user" if message.type == "human" else "assistant"
        return {"role": role, "content": message.content}
    else:
        # Fallback, treat as user message
        return {"role": "user", "content": str(message)}

def format_messages(messages):
    """Format a list of messages to consistent structure.
    
    Args:
        messages (list): List of messages to format
        
    Returns:
        list: List of formatted message dictionaries
    """
    return [format_message(msg) for msg in messages]

def get_last_message_content(messages):
    """Extract content from the last message.
    
    Args:
        messages (list): List of messages
        
    Returns:
        str: Content of the last message
    """
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return last_msg.get("content", "")
    elif hasattr(last_msg, "content"):
        return last_msg.content
    else:
        return str(last_msg)