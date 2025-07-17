import uuid
import pytest
from unittest import mock
from ataraxai.praxis.modules.chat.chat_context_manager import ChatContextManager

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
def chat_context_manager(db_manager):
    # Patch AutoTokenizer to avoid loading real models
    with mock.patch("ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer") as MockTokenizer:
        instance = MockTokenizer.from_pretrained.return_value
        instance.encode.side_effect = lambda text: list(range(len(text.split())))
        yield ChatContextManager(db_manager, model_name="gpt2")

def test_add_message(db_manager, chat_context_manager):
    session_id = uuid.uuid4()
    chat_context_manager.add_message(session_id, "user", "Hello!")
    assert len(db_manager.get_messages_for_session(session_id)) == 1
    msg = db_manager.get_messages_for_session(session_id)[0]
    assert msg.role == "user"
    assert msg.content == "Hello!"

def test_get_messages_for_session(db_manager, chat_context_manager):
    session_id = uuid.uuid4()
    db_manager.add_message(session_id, "user", "Hi")
    db_manager.add_message(session_id, "assistant", "Hello!")
    messages = chat_context_manager.get_messages_for_session(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[0]["content"] == "Hi"
    assert messages[1]["content"] == "Hello!"
    assert "timestamp" in messages[0]

def test_get_formatted_context_for_model_truncates(db_manager):
    with mock.patch("ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer") as MockTokenizer:
        instance = MockTokenizer.from_pretrained.return_value
        instance.encode.side_effect = lambda text: list(range(len(text.split())))
        chat_context_manager = ChatContextManager(db_manager, model_name="gpt2")
        chat_context_manager.max_tokens = 5  # Small limit for test

        session_id = uuid.uuid4()
        db_manager.add_message(session_id, "user", "one two")
        db_manager.add_message(session_id, "assistant", "three four")
        db_manager.add_message(session_id, "user", "five six seven")

        context = chat_context_manager.get_formatted_context_for_model(session_id)
        assert isinstance(context, list)
        total_tokens = sum(len(instance.encode(m["content"])) for m in context)
        assert total_tokens <= chat_context_manager.max_tokens

def test_get_formatted_context_for_model_no_tokenizer(db_manager):
    with mock.patch("ataraxai.praxis.modules.chat.chat_context_manager.AutoTokenizer.from_pretrained", side_effect=Exception):
        chat_context_manager = ChatContextManager(db_manager, model_name="gpt2")
        session_id = uuid.uuid4()
        db_manager.add_message(session_id, "user", "one two three")
        db_manager.add_message(session_id, "assistant", "four five six seven")
        context = chat_context_manager.get_formatted_context_for_model(session_id)
        assert isinstance(context, list)
        assert all("role" in m and "content" in m and "timestamp" in m for m in context)