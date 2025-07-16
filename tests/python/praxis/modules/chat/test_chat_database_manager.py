import uuid
import pytest
from pathlib import Path
from peewee import SqliteDatabase

from ataraxai.praxis.modules.chat.chat_database_manager import (
    ChatDatabaseManager,
    DatabaseError,
    NotFoundError,
)

from ataraxai.praxis.modules.chat.chat_database_manager import (
    ChatDatabaseManager,
    DatabaseError,
    NotFoundError,
    db,
)


@pytest.fixture
def db_manager(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db.init(str(db_path))

    manager = ChatDatabaseManager(db_path)

    yield manager

    if not db.is_closed():
        db.close()


def test_get_session_success(db_manager):
    project = db_manager.create_project("Test Project", "desc")
    session = db_manager.create_session(project.id, "Test Session")
    retrieved = db_manager.get_session(session.id)
    assert retrieved.id == session.id
    assert retrieved.title == "Test Session"
    assert retrieved.project.id == project.id


def test_get_session_not_found(db_manager):
    fake_id = uuid.uuid4()
    with pytest.raises(NotFoundError):
        db_manager.get_session(fake_id)


def test_get_session_invalid_id_type(db_manager):
    with pytest.raises(Exception):
        db_manager.get_session("not-a-uuid")


def test_create_and_get_message(db_manager):
    project = db_manager.create_project("Project for Message", "desc")
    session = db_manager.create_session(project.id, "Session for Message")
    message = db_manager.add_message(session.id, "user", "Hello, world!")
    retrieved = db_manager.get_message(message.id)
    assert retrieved.id == message.id
    assert retrieved.role == "user"
    assert retrieved.content == "Hello, world!"
    assert retrieved.session.id == session.id


def test_get_message_not_found(db_manager):
    fake_id = uuid.uuid4()
    with pytest.raises(NotFoundError):
        db_manager.get_message(fake_id)


def test_add_message_invalid_role(db_manager):
    project = db_manager.create_project("Project Invalid Role", "desc")
    session = db_manager.create_session(project.id, "Session Invalid Role")
    with pytest.raises(Exception):
        db_manager.add_message(session.id, "invalid_role", "Some content")


def test_add_message_empty_content(db_manager):
    project = db_manager.create_project("Project Empty Content", "desc")
    session = db_manager.create_session(project.id, "Session Empty Content")
    with pytest.raises(Exception):
        db_manager.add_message(session.id, "user", "")


def test_update_project(db_manager):
    project = db_manager.create_project("Update Project", "desc")
    updated = db_manager.update_project(
        project.id, name="Updated Name", description="Updated desc"
    )
    assert updated.name == "Updated Name"
    assert updated.description == "Updated desc"


def test_delete_project(db_manager):
    project = db_manager.create_project("Delete Project", "desc")
    result = db_manager.delete_project(project.id)
    assert result is True
    with pytest.raises(NotFoundError):
        db_manager.get_project(project.id)


def test_get_conversation_history(db_manager):
    project = db_manager.create_project("History Project", "desc")
    session = db_manager.create_session(project.id, "History Session")
    db_manager.add_message(session.id, "user", "Hi")
    db_manager.add_message(session.id, "assistant", "Hello!")
    history = db_manager.get_conversation_history(session.id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_search_across_project(db_manager):
    project = db_manager.create_project("Search Project", "desc")
    session = db_manager.create_session(project.id, "Search Session")
    db_manager.add_message(session.id, "user", "Find me")
    db_manager.add_message(session.id, "assistant", "Not this one")
    results = db_manager.search_across_project(project.id, "Find")
    assert "sessions" in results
    assert len(results["sessions"]) == 1
    assert results["sessions"][0]["messages"][0]["content"] == "Find me"


def test_update_message(db_manager):
    project = db_manager.create_project("UpdateMsg Project", "desc")
    session = db_manager.create_session(project.id, "UpdateMsg Session")
    message = db_manager.add_message(session.id, "user", "Original")
    updated = db_manager.update_message(message.id, role="assistant", content="Changed")
    assert updated.role == "assistant"
    assert updated.content == "Changed"


def test_delete_message(db_manager):
    project = db_manager.create_project("DeleteMsg Project", "desc")
    session = db_manager.create_session(project.id, "DeleteMsg Session")
    message = db_manager.add_message(session.id, "user", "To be deleted")
    result = db_manager.delete_message(message.id)
    assert result is True
    with pytest.raises(NotFoundError):
        db_manager.get_message(message.id)


def test_create_project_duplicate_name(db_manager):
    db_manager.create_project("Duplicate Project", "desc1")
    with pytest.raises(Exception):
        db_manager.create_project("Duplicate Project", "desc2")


def test_update_project_to_duplicate_name(db_manager):
    p1 = db_manager.create_project("Proj1", "desc1")
    p2 = db_manager.create_project("Proj2", "desc2")
    with pytest.raises(Exception):
        db_manager.update_project(p2.id, name="Proj1")


def test_create_session_empty_title(db_manager):
    project = db_manager.create_project("Sess Project", "desc")
    with pytest.raises(Exception):
        db_manager.create_session(project.id, "")


def test_create_session_project_not_found(db_manager):
    fake_id = uuid.uuid4()
    with pytest.raises(NotFoundError):
        db_manager.create_session(fake_id, "Should Fail")


def test_update_session_title(db_manager):
    project = db_manager.create_project("Session Update", "desc")
    session = db_manager.create_session(project.id, "Old Title")
    updated = db_manager.update_session(session.id, "New Title")
    assert updated.title == "New Title"


def test_update_session_empty_title(db_manager):
    project = db_manager.create_project("Session Empty", "desc")
    session = db_manager.create_session(project.id, "Title")
    with pytest.raises(Exception):
        db_manager.update_session(session.id, "")


def test_delete_session(db_manager):
    project = db_manager.create_project("DelSess Project", "desc")
    session = db_manager.create_session(project.id, "DelSess")
    result = db_manager.delete_session(session.id)
    assert result is True
    with pytest.raises(NotFoundError):
        db_manager.get_session(session.id)


def test_add_message_session_not_found(db_manager):
    fake_id = uuid.uuid4()
    with pytest.raises(NotFoundError):
        db_manager.add_message(fake_id, "user", "Hello")


def test_update_message_invalid_role(db_manager):
    project = db_manager.create_project("UpdateMsgRole Project", "desc")
    session = db_manager.create_session(project.id, "UpdateMsgRole Session")
    message = db_manager.add_message(session.id, "user", "Role test")
    with pytest.raises(Exception):
        db_manager.update_message(message.id, role="badrole")


def test_update_message_empty_content(db_manager):
    project = db_manager.create_project("UpdateMsgContent Project", "desc")
    session = db_manager.create_session(project.id, "UpdateMsgContent Session")
    message = db_manager.add_message(session.id, "user", "Content test")
    with pytest.raises(Exception):
        db_manager.update_message(message.id, content="")


def test_get_project_summary(db_manager):
    project = db_manager.create_project("Summary Project", "desc")
    session = db_manager.create_session(project.id, "Summary Session")
    db_manager.add_message(session.id, "user", "msg1")
    db_manager.add_message(session.id, "assistant", "msg2")
    summary = db_manager.get_project_summary(project.id)
    assert summary["project"]["name"] == "Summary Project"
    assert summary["stats"]["session_count"] == 1
    assert summary["stats"]["message_count"] == 2


def test_search_projects(db_manager):
    db_manager.create_project("Alpha", "desc1")
    db_manager.create_project("Beta", "desc2")
    results = db_manager.project_service.search_projects("Alpha")
    assert any("Alpha" in p.name for p in results)


def test_search_messages(db_manager):
    project = db_manager.create_project("SearchMsg Project", "desc")
    session = db_manager.create_session(project.id, "SearchMsg Session")
    db_manager.add_message(session.id, "user", "Find this message")
    db_manager.add_message(session.id, "assistant", "Other content")
    found = db_manager.message_service.search_messages(session.id, "Find")
    assert len(found) == 1
    assert "Find this message" in found[0].content


def test_transaction_context_manager(db_manager):
    # Should not raise
    with db_manager.transaction():
        project = db_manager.create_project("Tx Project", "desc")
        assert project.name == "Tx Project"


def test_close_and_context_manager(tmp_path, monkeypatch):
    db_path = tmp_path / "ctx_test.db"
    monkeypatch.setattr(
        "ataraxai.app_logic.modules.chat.chat_database_manager.db",
        SqliteDatabase(str(db_path)),
    )
    dbm = ChatDatabaseManager(db_path)
    dbm.close()
    with ChatDatabaseManager(db_path) as db:
        assert db is not None
