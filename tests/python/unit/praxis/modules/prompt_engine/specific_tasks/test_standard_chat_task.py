import pytest
from ataraxai.praxis.modules.prompt_engine.specific_tasks.standard_chat_task import StandardChatTask
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_dependencies():
    chat_context = MagicMock()
    rag_manager = MagicMock()
    prompt_manager = MagicMock()
    core_ai_service_manager = MagicMock()
    context_manager = MagicMock()

    chat_context.get_messages_for_session.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]
    context_manager.get_context.return_value = ["Relevant chunk 1", "Relevant chunk 2"]
    prompt_manager.load_template.return_value = "Prompt template"
    prompt_manager.build_prompt_within_limit.return_value = "Final prompt"
    core_ai_service_manager.get_llama_cpp_model_context_size.return_value = 2048
    core_ai_service_manager.process_prompt = AsyncMock(
        return_value="This is a response."
    )

    rag_manager.rag_config_manager.config = {}

    return {
        "chat_context": chat_context,
        "rag_manager": rag_manager,
        "prompt_manager": prompt_manager,
        "core_ai_service_manager": core_ai_service_manager,
        "context_manager": context_manager,
    }

@pytest.mark.asyncio
async def test_execute_returns_assistant_response(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "What is AI?",
        "session_id": "session123"
    }

    response = await task.execute(processed_input, mock_dependencies)

    assert response == "This is a response."
    mock_dependencies["chat_context"].add_message.assert_any_call(
        "session123", role="user", content="What is AI?"
    )
    mock_dependencies["chat_context"].add_message.assert_any_call(
        "session123", role="assistant", content="This is a response."
    )


@pytest.mark.asyncio
async def test_execute_handles_empty_model_response(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "What is AI?",
        "session_id": "session123"
    }
    mock_dependencies["core_ai_service_manager"].process_prompt.return_value = "   "

    response = await task.execute(processed_input, mock_dependencies)

    assert response == "I'm sorry, I couldn't generate a response."
    mock_dependencies["chat_context"].add_message.assert_any_call(
        "session123", role="assistant", content="I'm sorry, I couldn't generate a response."
    )

@pytest.mark.asyncio
async def test_execute_strips_assistant_prefix(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "Tell me a joke.",
        "session_id": "session456"
    }
    mock_dependencies["core_ai_service_manager"].process_prompt.return_value = "assistant: Here is a joke."

    response = await task.execute(processed_input, mock_dependencies)

    assert response == "Here is a joke."

@pytest.mark.asyncio
async def test_execute_includes_rag_context_and_prompt_template(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "Explain quantum computing.",
        "session_id": "session789"
    }

    response = await task.execute(processed_input, mock_dependencies)

    mock_dependencies["context_manager"].get_context.assert_called_once_with(
        context_key="relevant_document_chunks", user_inputs="Explain quantum computing."
    )
    mock_dependencies["prompt_manager"].load_template.assert_called_once_with("standard_chat")
    mock_dependencies["prompt_manager"].build_prompt_within_limit.assert_called_once()
    mock_dependencies["core_ai_service_manager"].process_prompt.assert_called_once_with("Final prompt")
    assert response == "This is a response."

@pytest.mark.asyncio
async def test_execute_handles_no_assistant_prefix(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "No prefix?",
        "session_id": "session999"
    }
    mock_dependencies["core_ai_service_manager"].process_prompt.return_value = "Just a plain response."

    response = await task.execute(processed_input, mock_dependencies)

    assert response == "Just a plain response."
    mock_dependencies["chat_context"].add_message.assert_any_call(
        "session999", role="assistant", content="Just a plain response."
    )

@pytest.mark.asyncio
async def test_execute_handles_multiple_assistant_prefixes(mock_dependencies):
    task = StandardChatTask()
    processed_input = {
        "user_query": "Multiple prefixes?",
        "session_id": "session1000"
    }
    mock_dependencies["core_ai_service_manager"].process_prompt.return_value = "assistant: assistant: Nested response."

    response = await task.execute(processed_input, mock_dependencies)

    assert response == "Nested response."
    mock_dependencies["chat_context"].add_message.assert_any_call(
        "session1000", role="assistant", content="Nested response."
    )
