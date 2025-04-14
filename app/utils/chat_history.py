def build_chat_history(messages):
    """
    Build a formatted string representation of chat history from message list.
    
    Args:
        messages (list): List of message objects
        
    Returns:
        str: Formatted chat history
    """
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