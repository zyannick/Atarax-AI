import pytest
from unittest.mock import MagicMock
from ataraxai.app_logic.modules.prompt_engine.context_manager import ContextManager
import re
from ataraxai.app_logic.modules.prompt_engine.context_manager import TaskContext

@pytest.fixture
def mock_rag_manager():
    return MagicMock()

@pytest.fixture
def config():
    return {
        "rag": {"top_k": 5}
    }

@pytest.fixture
def context_manager(config, mock_rag_manager):
    return ContextManager(config, mock_rag_manager)

def test_get_relevant_document_chunks_returns_empty_when_no_query(context_manager):
    user_inputs = {}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == []

def test_get_relevant_document_chunks_returns_empty_when_query_is_none(context_manager):
    user_inputs = {"query": None}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == []

def test_get_relevant_document_chunks_returns_empty_when_query_results_is_none(context_manager, mock_rag_manager):
    user_inputs = {"query": "test"}
    mock_rag_manager.query_knowledge.return_value = None
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == []

def test_get_relevant_document_chunks_returns_empty_when_documents_missing(context_manager, mock_rag_manager):
    user_inputs = {"query": "test"}
    mock_rag_manager.query_knowledge.return_value = {}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == []

def test_get_relevant_document_chunks_returns_empty_when_documents_empty(context_manager, mock_rag_manager):
    user_inputs = {"query": "test"}
    mock_rag_manager.query_knowledge.return_value = {"documents": []}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == []

def test_get_relevant_document_chunks_returns_first_document_list(context_manager, mock_rag_manager):
    user_inputs = {"query": "test"}
    documents = [["chunk1", "chunk2", "chunk3"], ["chunk4"]]
    mock_rag_manager.query_knowledge.return_value = {"documents": documents}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == ["chunk1", "chunk2", "chunk3"]
    mock_rag_manager.query_knowledge.assert_called_once_with(query_text="test", n_results=5)

def test_get_relevant_document_chunks_uses_default_top_k(context_manager, mock_rag_manager):
    context_manager.config.pop("rag", None)
    user_inputs = {"query": "test"}
    documents = [["chunkA"]]
    mock_rag_manager.query_knowledge.return_value = {"documents": documents}
    result = context_manager._get_relevant_document_chunks(user_inputs)
    assert result == ["chunkA"]
    mock_rag_manager.query_knowledge.assert_called_once_with(query_text="test", n_results=3)

def test_get_current_date_format(context_manager):
    date_str = context_manager._get_current_date()
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", date_str)

def test_get_calendar_events_today_returns_empty_list(context_manager):
    events = context_manager._get_calendar_events_today()
    assert isinstance(events, list)
    assert events == []

def test_get_file_content_reads_file(tmp_path, context_manager):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")
    content = context_manager._get_file_content(str(file_path))
    assert content == "hello world"

def test_get_file_content_file_not_found(context_manager):
    content = context_manager._get_file_content("nonexistent_file.txt")
    assert content is None

def test_get_file_content_handles_other_exceptions(monkeypatch, context_manager):
    def raise_exception(*args, **kwargs):
        raise Exception("Some error")
    monkeypatch.setattr("builtins.open", raise_exception)
    content = context_manager._get_file_content("anyfile.txt")
    assert content is None

def test_get_default_role_prompt_returns_persona_from_config():
    config = {
        "current_user_role": "admin",
        "roles": {"admin": {"default_persona_prompt_key": "admin_persona"}},
        "personas": {"admin_persona": "You are admin."}
    }
    cm = ContextManager(config, MagicMock())
    assert cm._get_default_role_prompt() == "You are admin."

def test_get_default_role_prompt_returns_default_when_missing():
    config = {}
    cm = ContextManager(config, MagicMock())
    assert cm._get_default_role_prompt() == "You are a helpful AI assistant."

def test_get_context_current_date(context_manager):
    result = context_manager.get_context("current_date")
    assert isinstance(result, str)

def test_get_context_default_role_prompt(context_manager):
    context_manager.config["current_user_role"] = "user"
    context_manager.config["roles"] = {"user": {"default_persona_prompt_key": "persona1"}}
    context_manager.config["personas"] = {"persona1": "Persona prompt"}
    result = context_manager.get_context("default_role_prompt")
    assert result == "Persona prompt"

def test_get_context_relevant_document_chunks(context_manager, mock_rag_manager):
    user_inputs = {"query": {"query": "test"}}
    mock_rag_manager.query_knowledge.return_value = {"documents": [["doc1"]]}
    result = context_manager.get_context("relevant_document_chunks", user_inputs=user_inputs)
    assert result == ["doc1"]

def test_get_context_user_calendar_today(context_manager):
    result = context_manager.get_context("user_calendar_today")
    assert result == []

def test_get_context_file_content(tmp_path, context_manager):
    file_path = tmp_path / "file.txt"
    file_path.write_text("abc", encoding="utf-8")
    user_inputs = {"file_path": str(file_path)}
    result = context_manager.get_context("file_content", user_inputs=user_inputs)
    assert result == "abc"

def test_get_context_returns_none_for_unknown_key(context_manager):
    result = context_manager.get_context("unknown_key")
    assert result is None

def test_task_context_add_to_history_and_summary():
    tc = TaskContext("query1")
    tc.add_to_history("entry1")
    assert tc.user_history == ["entry1"]
    summary = tc.get_context_summary()
    assert "query1" in summary and "entry1" in summary
