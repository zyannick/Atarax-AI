from typing_extensions import Optional


def create_prompt(
    user_query: str,
    system_message: Optional[str] = None,
    conversation_history: Optional[list[str]] = None,
) -> str:
    prompt_parts = []

    if system_message:
        prompt_parts.append(f"System: {system_message.strip()}")

    prompt_parts.append(f"User: {user_query.strip()}")
    prompt_parts.append("Assistant:")

    return "\n".join(prompt_parts)
