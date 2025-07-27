import pytest
from unittest import mock
from ataraxai.praxis.modules.prompt_engine.specific_tasks.standard_chat_task import (
    StandardChatTask,
)


@pytest.fixture
def mock_dependencies():
    chat_context = mock.Mock()
    prompt_manager = mock.Mock()
    rag_manager = mock.Mock()
    core_ai_service = mock.Mock()
    return {
        "chat_context": chat_context,
        "prompt_manager": prompt_manager,
        "rag_manager": rag_manager,
        "core_ai_service": core_ai_service,
        "generation_params": {"temperature": 0.7},
    }


@pytest.fixture
def processed_input():
    return {"user_query": "What is AI?", "session_id": "session123"}


@pytest.fixture
def context():
    return mock.Mock()


def test_execute_adds_user_and_assistant_messages(
    processed_input, context, mock_dependencies
):
    chat_context = mock_dependencies["chat_context"]
    prompt_manager = mock_dependencies["prompt_manager"]
    rag_manager = mock_dependencies["rag_manager"]
    core_ai_service = mock_dependencies["core_ai_service"]

    chat_context.get_messages_for_session.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    rag_manager.query_knowledge.return_value = {
        "documents": [["AI is artificial intelligence."]]
    }
    prompt_manager.load_template.return_value = "Prompt with context"
    core_ai_service.process_prompt.return_value = (
        "AI stands for Artificial Intelligence."
    )

    task = StandardChatTask()
    result = task.execute(processed_input, context, mock_dependencies)

    chat_context.add_message.assert_any_call(
        "session123", role="user", content="What is AI?"
    )
    chat_context.add_message.assert_any_call(
        "session123", role="assistant", content="AI stands for Artificial Intelligence."
    )

    prompt_manager.load_template.assert_called_once_with(
        "main_chat",
        history=chat_context.get_messages_for_session.return_value,
        context="AI is artificial intelligence.",
        query="What is AI?",
    )

    rag_manager.query_knowledge.assert_called_once_with(
        query_text="What is AI?", n_results=3
    )

    core_ai_service.process_prompt.assert_called_once_with(
        "Prompt with context", {"temperature": 0.7}
    )

    assert result == "AI stands for Artificial Intelligence."


def test_execute_handles_no_rag_documents(processed_input, context, mock_dependencies):
    rag_manager = mock_dependencies["rag_manager"]
    prompt_manager = mock_dependencies["prompt_manager"]
    core_ai_service = mock_dependencies["core_ai_service"]

    rag_manager.query_knowledge.return_value = {"documents": []}
    prompt_manager.load_template.return_value = "Prompt with no context"
    core_ai_service.process_prompt.return_value = "Sorry, I don't know."

    task = StandardChatTask()
    result = task.execute(processed_input, context, mock_dependencies)

    prompt_manager.load_template.assert_called_once_with(
        "main_chat",
        history=mock_dependencies["chat_context"].get_messages_for_session.return_value,
        context="No relevant documents found.",
        query="What is AI?",
    )
    assert result == "Sorry, I don't know."
