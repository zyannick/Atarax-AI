import tempfile
import shutil
from pathlib import Path
import pytest
from ataraxai.praxis.modules.chat import chat_database_manager as cdm
import uuid



@pytest.fixture(scope="function")
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_chat.db"
    manager = cdm.ChatDatabaseManager(db_path)
    yield manager
    manager.project_service.model._meta.database.close()
    shutil.rmtree(temp_dir)

def test_create_and_get_project(temp_db):
    project = temp_db.create_project("Test Project", "A test project")
    assert project.name == "Test Project"
    assert project.description == "A test project"
    fetched = temp_db.get_project(project.id)
    assert fetched.id == project.id
    assert fetched.name == "Test Project"

def test_create_project_empty_name_raises(temp_db):
    with pytest.raises(cdm.DatabaseError):
        temp_db.create_project("", "desc")

def test_create_duplicate_project_raises(temp_db):
    temp_db.create_project("Unique", "desc")
    with pytest.raises(cdm.DatabaseError):
        temp_db.create_project("Unique", "desc2")

def test_update_project(temp_db):
    project = temp_db.create_project("ToUpdate", "desc")
    updated = temp_db.update_project(project.id, name="Updated", description="New desc")
    assert updated.name == "Updated"
    assert updated.description == "New desc"

def test_delete_project(temp_db):
    project = temp_db.create_project("ToDelete", "desc")
    assert temp_db.delete_project(project.id) is True
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_project(project.id)

def test_create_and_get_session(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Session 1")
    assert session.title == "Session 1"
    fetched = temp_db.get_session(session.id)
    assert fetched.id == session.id

def test_create_session_empty_title_raises(temp_db):
    project = temp_db.create_project("Proj", "desc")
    with pytest.raises(cdm.DatabaseError):
        temp_db.create_session(project.id, "")

def test_update_session(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Old Title")
    updated = temp_db.update_session(session.id, "New Title")
    assert updated.title == "New Title"

def test_delete_session(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "ToDelete")
    assert temp_db.delete_session(session.id) is True
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_session(session.id)

def test_add_and_get_message(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    msg = temp_db.add_message(session.id, "user", "Hello!")
    assert msg.role == "user"
    assert msg.content == "Hello!"
    fetched = temp_db.get_message(msg.id)
    assert fetched.id == msg.id

def test_add_message_empty_content_raises(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    with pytest.raises(cdm.DatabaseError):
        temp_db.add_message(session.id, "user", "")

def test_add_message_invalid_role_raises(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    with pytest.raises(cdm.DatabaseError):
        temp_db.add_message(session.id, "invalidrole", "Content")

def test_update_message(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    msg = temp_db.add_message(session.id, "user", "Hi")
    updated = temp_db.update_message(msg.id, role="assistant", content="Hello back")
    assert updated.role == "assistant"
    assert updated.content == "Hello back"

def test_delete_message(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    msg = temp_db.add_message(session.id, "user", "Bye")
    assert temp_db.delete_message(msg.id) is True
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_message(msg.id)

def test_get_conversation_history(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    temp_db.add_message(session.id, "user", "Hi")
    temp_db.add_message(session.id, "assistant", "Hello!")
    history = temp_db.get_conversation_history(session.id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_list_projects(temp_db):
    temp_db.create_project("A", "desc")
    temp_db.create_project("B", "desc")
    projects = temp_db.list_projects()
    assert len(projects) >= 2

def test_get_sessions_for_project(temp_db):
    project = temp_db.create_project("Proj", "desc")
    temp_db.create_session(project.id, "S1")
    temp_db.create_session(project.id, "S2")
    sessions = temp_db.get_sessions_for_project(project.id)
    assert len(sessions) == 2

def test_get_messages_for_session(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    temp_db.add_message(session.id, "user", "Hi")
    temp_db.add_message(session.id, "assistant", "Hello!")
    messages = temp_db.get_messages_for_session(session.id)
    assert len(messages) == 2
    

def test_delete_project_also_deletes_sessions_and_messages(temp_db):
    project = temp_db.create_project("Cascade", "desc")
    session = temp_db.create_session(project.id, "Cascade Session")
    msg = temp_db.add_message(session.id, "user", "Cascade message")
    assert temp_db.delete_project(project.id) is True
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_project(project.id)
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_session(session.id)
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_message(msg.id)

def test_delete_session_also_deletes_messages(temp_db):
    project = temp_db.create_project("Cascade2", "desc")
    session = temp_db.create_session(project.id, "Cascade2 Session")
    msg = temp_db.add_message(session.id, "user", "Cascade2 message")
    assert temp_db.delete_session(session.id) is True
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_session(session.id)
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_message(msg.id)

def test_update_message_invalid_role_raises(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    msg = temp_db.add_message(session.id, "user", "Hi")
    with pytest.raises(cdm.DatabaseError):
        temp_db.update_message(msg.id, role="notarole")

def test_update_message_empty_content_raises(temp_db):
    project = temp_db.create_project("Proj", "desc")
    session = temp_db.create_session(project.id, "Sess")
    msg = temp_db.add_message(session.id, "user", "Hi")
    with pytest.raises(cdm.DatabaseError):
        temp_db.update_message(msg.id, content="")

def test_get_project_not_found_raises(temp_db):
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_project(uuid.uuid4())

def test_get_session_not_found_raises(temp_db):
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_session(uuid.uuid4())

def test_get_message_not_found_raises(temp_db):
    with pytest.raises(cdm.NotFoundError):
        temp_db.get_message(uuid.uuid4())


def test_get_conversation_history_returns_correct_format_and_order(temp_db):
    project = temp_db.create_project("HistoryProj", "desc")
    session = temp_db.create_session(project.id, "HistorySession")
    msg1 = temp_db.add_message(session.id, "user", "First message")
    msg2 = temp_db.add_message(session.id, "assistant", "Second message")
    history = temp_db.get_conversation_history(session.id)
    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0]["id"] == str(msg1.id)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == b"First message"
    assert "timestamp" in history[0]
    assert history[1]["id"] == str(msg2.id)
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == b"Second message"
    assert "timestamp" in history[1]

def test_get_conversation_history_empty(temp_db):
    project = temp_db.create_project("EmptyProj", "desc")
    session = temp_db.create_session(project.id, "EmptySession")
    history = temp_db.get_conversation_history(session.id)
    assert isinstance(history, list)
    assert len(history) == 0


def test_get_project_summary_returns_dict(temp_db):
    project = temp_db.create_project("SummaryProj", "desc")
    session1 = temp_db.create_session(project.id, "Session1")
    session2 = temp_db.create_session(project.id, "Session2")
    temp_db.add_message(session1.id, "user", "Hello 1")
    temp_db.add_message(session1.id, "assistant", "Hi 1")
    temp_db.add_message(session2.id, "user", "Hello 2")
    summary = temp_db.get_project_summary(project.id)
    assert isinstance(summary, dict)
    assert summary["project"]["id"] == str(project.id)
    assert summary["project"]["name"] == "SummaryProj"
    assert summary["project"]["description"] == "desc"
    assert summary["stats"]["session_count"] == 2
    assert summary["stats"]["message_count"] == 3

def test_get_project_summary_no_sessions(temp_db):
    project = temp_db.create_project("NoSessProj", "desc")
    summary = temp_db.get_project_summary(project.id)
    assert summary["stats"]["session_count"] == 0
    assert summary["stats"]["message_count"] == 0

def test_get_project_summary_invalid_project_raises(temp_db):
    with pytest.raises(cdm.DatabaseError):
        temp_db.get_project_summary(uuid.uuid4())


def test_project_str_repr(temp_db):
    project = temp_db.create_project("StrTest", "desc")
    s = str(project)
    assert "Project(id=" in s and "StrTest" in s

def test_session_str_repr(temp_db):
    project = temp_db.create_project("StrTest", "desc")
    session = temp_db.create_session(project.id, "SessionStr")
    s = str(session)
    assert "ChatSession(id=" in s and "SessionStr" in s

def test_message_str_repr(temp_db):
    project = temp_db.create_project("StrTest", "desc")
    session = temp_db.create_session(project.id, "SessionStr")
    msg = temp_db.add_message(session.id, "user", "A" * 60)
    s = str(msg)
    assert "Message(id=" in s and "user" in s and "..." in s

def test_project_getters(temp_db):
    project = temp_db.create_project("GetterProj", "desc")
    assert isinstance(project.get_id(), uuid.UUID)
    assert isinstance(project.get_created_at(), project.created_at.__class__)
    assert isinstance(project.get_updated_at(), project.updated_at.__class__)
    assert project.get_name() == "GetterProj"
    assert project.get_description() == "desc"

def test_session_getters(temp_db):
    project = temp_db.create_project("GetterProj", "desc")
    session = temp_db.create_session(project.id, "SessTitle")
    assert isinstance(session.get_id(), uuid.UUID)
    assert isinstance(session.get_created_at(), session.created_at.__class__)
    assert isinstance(session.get_updated_at(), session.updated_at.__class__)
    assert session.get_title() == "SessTitle"
    assert session.get_project_id() == project.id

def test_message_getters(temp_db):
    project = temp_db.create_project("GetterProj", "desc")
    session = temp_db.create_session(project.id, "SessTitle")
    msg = temp_db.add_message(session.id, "user", b"Hello bytes")
    assert isinstance(msg.get_id(), uuid.UUID)
    assert isinstance(msg.get_session_id(), uuid.UUID)
    assert msg.get_role() == "user"
    assert isinstance(msg.get_content(), bytes)
    assert isinstance(msg.get_timestamp(), msg.date_time.__class__)
    assert isinstance(msg.get_date_time(), msg.date_time.__class__)

def test_transaction_context_manager_success(temp_db):
    with temp_db.transaction():
        project = temp_db.create_project("TxProj", "desc")
        assert project.name == "TxProj"

def test_transaction_context_manager_raises(temp_db):
    with pytest.raises(Exception):
        with temp_db.transaction():
            raise Exception("fail in tx")

def test_search_projects(temp_db):
    temp_db.create_project("Alpha", "desc1")
    temp_db.create_project("Beta", "desc2")
    found = temp_db.project_service.search_projects("Alpha")
    assert any(p.name == "Alpha" for p in found)
    found2 = temp_db.project_service.search_projects("desc2")
    assert any(p.name == "Beta" for p in found2)










