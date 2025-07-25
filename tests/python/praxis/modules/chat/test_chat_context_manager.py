import uuid
import pytest
from unittest import mock
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager


@pytest.fixture
def mock_db_manager():
    return mock.Mock()

@pytest.fixture
def mock_vault_manager():
    vm = mock.Mock()
    vm.encrypt.side_effect = lambda b: b"encrypted_" + b
    vm.decrypt.side_effect = lambda b: b.replace(b"encrypted_", b"")
    return vm

@pytest.fixture
def chat_context_manager(mock_db_manager, mock_vault_manager):
    with mock.patch("ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer") as mock_tokenizer_cls:
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.side_effect = lambda s: list(range(len(s.split())))
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        yield ChatContextManager(mock_db_manager, mock_vault_manager, model_name="gpt2")

def test_add_message_encrypts_and_stores(chat_context_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    role = "user"
    content = "Hello, world!"
    chat_context_manager.add_message(session_id, role, content)
    encrypted_content = mock_vault_manager.encrypt(content.encode("utf-8"))
    mock_db_manager.add_message.assert_called_once_with(session_id, role, encrypted_content)

def test_get_messages_for_session_decrypts_and_returns(chat_context_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    mock_message = mock.Mock()
    mock_message.role = "assistant"
    mock_message.content = b"encrypted_Hi!"
    mock_message.date_time = "2024-06-01T12:00:00"
    mock_db_manager.get_messages_for_session.return_value = [mock_message]
    result = chat_context_manager.get_messages_for_session(session_id)
    assert result == [{
        "role": "assistant",
        "content": "Hi!",
        "date_time": "2024-06-01T12:00:00"
    }]
    mock_db_manager.get_messages_for_session.assert_called_once_with(session_id)

def test_get_formatted_on_token_limit(chat_context_manager, mock_db_manager, mock_vault_manager):
    session_id = uuid.uuid4()
    def make_msg(role, content, dt):
        m = mock.Mock()
        m.role = role
        m.content = b"encrypted_" + content.encode("utf-8")
        m.date_time = dt
        return m
    messages = [
        make_msg("user", "word " * 1000, "2024-06-01T10:00:00"),
        make_msg("assistant", "word " * 1000, "2024-06-01T11:00:00"),
        make_msg("user", "word " * 1000, "2024-06-01T12:00:00"),
    ]
    mock_db_manager.get_messages_for_session.return_value = messages
    chat_context_manager.max_tokens = 2000  
    result = chat_context_manager.get_formatted_context_for_model(session_id)
    assert len(result) == 2
    assert result[0]["date_time"] == "2024-06-01T11:00:00"
    assert result[1]["date_time"] == "2024-06-01T12:00:00"

def test_get_formatted_without_tokenizer(mock_db_manager, mock_vault_manager):
    with mock.patch("ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained", side_effect=Exception):
        ccm = ChatContextManager(mock_db_manager, mock_vault_manager, model_name="gpt2")
    session_id = uuid.uuid4()
    msg = mock.Mock()
    msg.role = "user"
    msg.content = b"encrypted_hello world"
    msg.date_time = "2024-06-01T13:00:00"
    mock_db_manager.get_messages_for_session.return_value = [msg]
    ccm.max_tokens = 1  
    result = ccm.get_formatted_context_for_model(session_id)
    assert result == []
