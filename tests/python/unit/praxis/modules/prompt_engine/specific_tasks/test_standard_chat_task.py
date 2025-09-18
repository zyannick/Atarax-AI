from unittest.mock import AsyncMock, MagicMock

import pytest

from ataraxai.praxis.modules.prompt_engine.specific_tasks.standard_chat_task import (
    StandardChatTask,
)


@pytest.mark.asyncio
async def test_execute_returns_assistant_response():
    task = StandardChatTask()
    processed_input = {"user_query": "Hello, how are you?", "session_id": "session123"}

    chat_context = AsyncMock()
    chat_context.get_messages_for_session.return_value = ["user: Hello, how are you?"]
    chat_context.add_message = AsyncMock()

    rag_manager = MagicMock()
    rag_manager.rag_config_manager.config = {"some": "config"}

    prompt_manager = MagicMock()
    prompt_manager.load_template.return_value = "Prompt template"
    prompt_manager.build_prompt_within_limit = AsyncMock(return_value="Final prompt")

    core_ai_service_manager = MagicMock()
    core_ai_service_manager.get_llama_cpp_model_context_size.return_value = 2048
    core_ai_service_manager.process_prompt = AsyncMock(
        return_value="assistant: I am fine, thank you!"
    )

    context_manager = MagicMock()
    context_manager.get_context = AsyncMock(
        return_value=["Relevant chunk 1", "Relevant chunk 2"]
    )

    dependencies = {
        "chat_context": chat_context,
        "rag_manager": rag_manager,
        "prompt_manager": prompt_manager,
        "core_ai_service_manager": core_ai_service_manager,
        "context_manager": context_manager,
    }

    response = await task.execute(processed_input, dependencies)

    assert response == "I am fine, thank you!"
    chat_context.add_message.assert_any_call(
        "session123", role="user", content="Hello, how are you?"
    )
    chat_context.add_message.assert_any_call(
        "session123", role="assistant", content="I am fine, thank you!"
    )
    prompt_manager.load_template.assert_called_once_with("standard_chat")
    prompt_manager.build_prompt_within_limit.assert_awaited()
    core_ai_service_manager.process_prompt.assert_awaited_with("Final prompt")


@pytest.mark.asyncio
async def test_execute_empty_model_response():
    task = StandardChatTask()
    processed_input = {"user_query": "What is the weather?", "session_id": "session456"}

    chat_context = AsyncMock()
    chat_context.get_messages_for_session.return_value = []
    chat_context.add_message = AsyncMock()

    rag_manager = MagicMock()
    rag_manager.rag_config_manager.config = {}

    prompt_manager = MagicMock()
    prompt_manager.load_template.return_value = "Prompt"
    prompt_manager.build_prompt_within_limit = AsyncMock(return_value="Prompt")

    core_ai_service_manager = MagicMock()
    core_ai_service_manager.get_llama_cpp_model_context_size.return_value = 1024
    core_ai_service_manager.process_prompt = AsyncMock(return_value="")

    context_manager = MagicMock()
    context_manager.get_context = AsyncMock(return_value=[])

    dependencies = {
        "chat_context": chat_context,
        "rag_manager": rag_manager,
        "prompt_manager": prompt_manager,
        "core_ai_service_manager": core_ai_service_manager,
        "context_manager": context_manager,
    }

    response = await task.execute(processed_input, dependencies)

    assert response == "I'm sorry, I couldn't generate a response."
    chat_context.add_message.assert_any_call(
        "session456",
        role="assistant",
        content="I'm sorry, I couldn't generate a response.",
    )


@pytest.mark.asyncio
async def test_execute_no_rag_results():
    task = StandardChatTask()
    processed_input = {"user_query": "Tell me a joke.", "session_id": "session789"}

    chat_context = AsyncMock()
    chat_context.get_messages_for_session.return_value = ["user: Tell me a joke."]
    chat_context.add_message = AsyncMock()

    rag_manager = MagicMock()
    rag_manager.rag_config_manager.config = {}

    prompt_manager = MagicMock()
    prompt_manager.load_template.return_value = "Joke prompt"
    prompt_manager.build_prompt_within_limit = AsyncMock(return_value="Joke prompt")

    core_ai_service_manager = MagicMock()
    core_ai_service_manager.get_llama_cpp_model_context_size.return_value = 512
    core_ai_service_manager.process_prompt = AsyncMock(
        return_value="assistant: Why did the chicken cross the road?"
    )

    context_manager = MagicMock()
    context_manager.get_context = AsyncMock(return_value=None)

    dependencies = {
        "chat_context": chat_context,
        "rag_manager": rag_manager,
        "prompt_manager": prompt_manager,
        "core_ai_service_manager": core_ai_service_manager,
        "context_manager": context_manager,
    }

    response = await task.execute(processed_input, dependencies)

    assert response == "Why did the chicken cross the road?"
    chat_context.add_message.assert_any_call(
        "session789", role="assistant", content="Why did the chicken cross the road?"
    )
