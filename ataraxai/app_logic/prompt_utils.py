def create_prompt(
    user_query: str, system_message: str = None, conversation_history=[]
) -> str:
    prompt_parts = []

    if system_message:
        prompt_parts.append(f"System: {system_message.strip()}")

    # prompt_parts.extend(conversation_history)

    prompt_parts.append(f"User: {user_query.strip()}")
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)
