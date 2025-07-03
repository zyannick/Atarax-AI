import uuid
import pytest
from unittest import mock
from ataraxai.app_logic.modules.chat.chat_context_manager import ChatContextManager


class DummyMessage:
    def __init__(self, role, content, timestamp):
        self.role = role
        self.content = content
        self.timestamp = timestamp


class DummyDBManager:
    def __init__(self):
        self.messages = {}

    def add_message(self, session_id, role, content):
        msg = DummyMessage(role, content, "2024-01-01T00:00:00Z")
        self.messages.setdefault(session_id, []).append(msg)

    def get_messages_for_session(self, session_id):
        return self.messages.get(session_id, [])


@pytest.fixture
def db_manager():
    return DummyDBManager()


@pytest.fixture
def session_id():
    return uuid.uuid4()


def test_add_message(db_manager, session_id):
    manager = ChatContextManager(db_manager)
    manager.add_message(session_id, "user", "Hello!")
    assert len(db_manager.get_messages_for_session(session_id)) == 1
    msg = db_manager.get_messages_for_session(session_id)[0]
    assert msg.role == "user"
    assert msg.content == "Hello!"


def test_get_messages_for_session(db_manager, session_id):
    db_manager.add_message(session_id, "user", "Hi")
    db_manager.add_message(session_id, "assistant", "Hello")
    manager = ChatContextManager(db_manager)
    messages = manager.get_messages_for_session(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_get_formatted_context_for_model_with_tokenizer(db_manager, session_id):
    db_manager.add_message(session_id, "user", "Hi")
    db_manager.add_message(session_id, "assistant", "Hello")
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.side_effect = lambda x: [0] * len(x.split())
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        manager = ChatContextManager(db_manager)
        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


def test_get_formatted_context_for_model_without_tokenizer(db_manager, session_id):
    db_manager.add_message(session_id, "user", "Hi there")
    db_manager.add_message(session_id, "assistant", "Hello again")
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained",
        side_effect=Exception(),
    ):
        manager = ChatContextManager(db_manager)
        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


def test_context_truncation(db_manager, session_id):
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.side_effect = lambda x: [0] * len(x.split())
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        manager = ChatContextManager(db_manager)
        manager.max_tokens = 3

        db_manager.add_message(session_id, "user", "one two")
        db_manager.add_message(session_id, "assistant", "three four")
        db_manager.add_message(session_id, "user", "five six")

        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 1
        assert context[0]["content"] == "five six"


def test_tokenizer_load_failure_prints_warning(db_manager, session_id, capsys):
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained",
        side_effect=Exception("fail"),
    ):
        manager = ChatContextManager(db_manager)
        out = capsys.readouterr().out
        assert "Warning: Could not load tokenizer" in out
        assert manager.tokenizer is None
        assert manager.max_tokens == 2048


def test_get_formatted_context_for_model_token_counting_with_tokenizer(
    db_manager, session_id
):
    db_manager.add_message(session_id, "user", "one two three")
    db_manager.add_message(session_id, "assistant", "four five six seven")
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.side_effect = lambda x: [0] * len(x.split())
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        manager = ChatContextManager(db_manager)
        manager.max_tokens = 5
        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 1
        assert context[0]["role"] == "assistant"
        assert context[0]["content"] == "four five six seven"


def test_get_formatted_context_for_model_token_counting_without_tokenizer(
    db_manager, session_id
):
    db_manager.add_message(session_id, "user", "one two three")
    db_manager.add_message(session_id, "assistant", "four five six seven")
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained",
        side_effect=Exception(),
    ):
        manager = ChatContextManager(db_manager)
        manager.max_tokens = 5
        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 1
        assert context[0]["role"] == "assistant"
        assert context[0]["content"] == "four five six seven"


def test_get_formatted_context_for_model_empty_session(db_manager, session_id):
    manager = ChatContextManager(db_manager)
    context = manager.get_formatted_context_for_model(session_id)
    assert context == []


def test_get_messages_for_session_empty(db_manager, session_id):
    manager = ChatContextManager(db_manager)
    messages = manager.get_messages_for_session(session_id)
    assert messages == []


def test_init_with_tokenizer_loads_tokenizer(db_manager):
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        mock_tokenizer = mock.Mock()
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        manager = ChatContextManager(db_manager, model_name="some-model")
        assert manager.tokenizer is mock_tokenizer
        assert manager.max_tokens == 4096


def test_init_without_tokenizer_sets_defaults(db_manager, capsys):
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained",
        side_effect=Exception(),
    ):
        manager = ChatContextManager(db_manager, model_name="bad-model")
        out = capsys.readouterr().out
        assert "Warning: Could not load tokenizer" in out
        assert manager.tokenizer is None
        assert manager.max_tokens == 2048


def test_add_message_calls_db_manager(db_manager, session_id):
    manager = ChatContextManager(db_manager)
    manager.add_message(session_id, "user", "test message")
    messages = db_manager.get_messages_for_session(session_id)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "test message"


def test_get_messages_for_session_returns_dicts(db_manager, session_id):
    db_manager.add_message(session_id, "user", "msg1")
    db_manager.add_message(session_id, "assistant", "msg2")
    manager = ChatContextManager(db_manager)
    result = manager.get_messages_for_session(session_id)
    assert isinstance(result, list)
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert "timestamp" in result[0]


def test_get_formatted_context_for_model_returns_all_when_under_limit(
    db_manager, session_id
):
    with mock.patch(
        "ataraxai.app_logic.modules.chat.chat_context_manager.AutoTokenizer"
    ) as MockTokenizer:
        mock_tokenizer = mock.Mock()
        mock_tokenizer.encode.side_effect = lambda x: [0] * len(x.split())
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        manager = ChatContextManager(db_manager)
        manager.max_tokens = 10
        db_manager.add_message(session_id, "user", "one two")
        db_manager.add_message(session_id, "assistant", "three four")
        context = manager.get_formatted_context_for_model(session_id)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"


def test_get_formatted_context_for_model_empty_returns_empty(db_manager, session_id):
    manager = ChatContextManager(db_manager)
    context = manager.get_formatted_context_for_model(session_id)
    assert context == []
