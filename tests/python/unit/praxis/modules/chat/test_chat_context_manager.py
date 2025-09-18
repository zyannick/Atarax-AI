import uuid
from typing import List
from unittest import mock

import pytest

from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager


@pytest.fixture
def mock_db_manager():
    return mock.AsyncMock()


@pytest.fixture
def mock_vault_manager():
    vm = mock.Mock()
    vm.encrypt.side_effect = lambda x: b"encrypted_" + x
    vm.decrypt.side_effect = lambda x: b"decrypted_" + x.replace(b"encrypted_", b"") # type: ignore
    return vm


@pytest.fixture
def chat_context_manager(
    mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    with mock.patch(
        "ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        MockTokenizer.from_pretrained.return_value.encode.side_effect = lambda x: [
            1
        ] * len(x.split()) # type: ignore
        return ChatContextManager(
            mock_db_manager, mock_vault_manager, model_name="gpt2"
        )


@pytest.mark.asyncio
async def test_add_message(
    chat_context_manager: ChatContextManager,
    mock_db_manager: mock.AsyncMock,
    mock_vault_manager: mock.Mock,
):
    session_id = uuid.uuid4()
    role = "user"
    content = "Hello, world!"
    await chat_context_manager.add_message(session_id, role, content)
    encrypted_content = mock_vault_manager.encrypt(content.encode("utf-8"))
    mock_db_manager.add_message.assert_awaited_once_with(
        session_id, role, encrypted_content
    )


@pytest.mark.asyncio
async def test_get_messages_for_session(
    chat_context_manager: ChatContextManager,
    mock_db_manager: mock.AsyncMock,
    mock_vault_manager: mock.Mock,
):
    session_id = uuid.uuid4()
    message_mock = mock.Mock()
    message_mock.role = "assistant"
    message_mock.content = b"encrypted_response"
    message_mock.date_time = "2024-06-01T12:00:00"
    mock_db_manager.get_messages_for_session.return_value = [message_mock]

    result = await chat_context_manager.get_messages_for_session(session_id)
    assert result == [
        {
            "role": "assistant",
            "content": "decrypted_response",
            "date_time": "2024-06-01T12:00:00",
        }
    ]


@pytest.mark.asyncio
async def test_get_formatted_context_for_model_truncates(
    chat_context_manager: ChatContextManager,
    mock_db_manager: mock.AsyncMock,
    mock_vault_manager: mock.Mock,
):
    session_id = uuid.uuid4()
    messages: List[mock.Mock] = []
    for i in range(5):
        msg = mock.Mock()
        msg.role = "user"
        msg.content = b"encrypted_" + f"message {i}".encode("utf-8")
        msg.date_time = f"2024-06-01T12:00:0{i}"
        messages.append(msg)
    mock_db_manager.get_messages_for_session.return_value = messages

    chat_context_manager.max_tokens = 3
    result = await chat_context_manager.get_formatted_context_for_model(session_id)
    assert len(result) == 1
    assert result[0]["content"] == "decrypted_message 4"


@pytest.mark.asyncio
async def test_get_formatted_context_for_model_no_tokenizer(
    mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    session_id = uuid.uuid4()
    msg = mock.Mock()
    msg.role = "user"
    msg.content = b"encrypted_test"
    msg.date_time = "2024-06-01T12:00:00"
    mock_db_manager.get_messages_for_session.return_value = [msg]

    manager = ChatContextManager(
        mock_db_manager, mock_vault_manager, model_name="nonexistent-model"
    )
    manager.tokenizer = None
    manager.max_tokens = 10
    result = await manager.get_formatted_context_for_model(session_id)
    assert result[0]["content"] == "decrypted_test"


@pytest.mark.asyncio
async def test_add_message_encrypts_and_stores(
    chat_context_manager: ChatContextManager, mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    session_id = uuid.uuid4()
    role = "assistant"
    content = "Test message"
    await chat_context_manager.add_message(session_id, role, content)
    encrypted_content = mock_vault_manager.encrypt(content.encode("utf-8"))
    mock_db_manager.add_message.assert_awaited_once_with(
        session_id, role, encrypted_content
    )


@pytest.mark.asyncio
async def test_get_messages_for_session_decrypts_content(
    chat_context_manager: ChatContextManager, mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    session_id = uuid.uuid4()
    msg = mock.Mock()
    msg.role = "user"
    msg.content = b"encrypted_hello"
    msg.date_time = "2024-06-01T12:00:00"
    mock_db_manager.get_messages_for_session.return_value = [msg]
    result = await chat_context_manager.get_messages_for_session(session_id)
    assert result == [
        {
            "role": "user",
            "content": "decrypted_hello",
            "date_time": "2024-06-01T12:00:00",
        }
    ]


@pytest.mark.asyncio
async def test_get_formatted_context_for_model_tokenizer_counts_tokens(
    chat_context_manager: ChatContextManager, mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    session_id = uuid.uuid4()
    msg1 = mock.Mock()
    msg1.role = "user"
    msg1.content = b"encrypted_first"
    msg1.date_time = "2024-06-01T12:00:00"
    msg2 = mock.Mock()
    msg2.role = "assistant"
    msg2.content = b"encrypted_second"
    msg2.date_time = "2024-06-01T12:01:00"
    mock_db_manager.get_messages_for_session.return_value = [msg1, msg2]
    chat_context_manager.max_tokens = 100
    result = await chat_context_manager.get_formatted_context_for_model(session_id)
    assert result[0]["content"] == "decrypted_first"
    assert result[1]["content"] == "decrypted_second"


@pytest.mark.asyncio
async def test_get_formatted_context_for_model_empty_message_skipped(
    chat_context_manager: ChatContextManager, mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    session_id = uuid.uuid4()
    msg1 = mock.Mock()
    msg1.role = "user"
    msg1.content = b"encrypted_"
    msg1.date_time = "2024-06-01T12:00:00"
    mock_vault_manager.decrypt.return_value = b""
    mock_db_manager.get_messages_for_session.return_value = [msg1]
    result = await chat_context_manager.get_formatted_context_for_model(session_id)
    assert result == []


@pytest.mark.asyncio
async def test_init_tokenizer_failure_sets_defaults(
    mock_db_manager: mock.AsyncMock, mock_vault_manager: mock.Mock
):
    with mock.patch(
        "ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained",
        side_effect=Exception,
    ):
        manager = ChatContextManager(
            mock_db_manager, mock_vault_manager, model_name="bad-model"
        )
        assert manager.tokenizer is None
        assert manager.max_tokens == 2048
