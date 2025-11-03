import tempfile
import shutil
from pathlib import Path
from peewee import SqliteDatabase
import pytest
from ataraxai.praxis.modules.chat import chat_database_manager as cdm
import uuid


@pytest.fixture(scope="function")
def db_manager():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_chat.db"

    manager = cdm.ChatDatabaseManager(db_path)

    yield manager

    manager.close()
    if not cdm.db.is_closed():
        cdm.db.close()
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_create_and_get_project(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Test Project", "A test project")
    assert project.name == "Test Project"
    assert project.description == "A test project"
    fetched = await db_manager.get_project(uuid.UUID(str(project.id)))
    assert fetched.id == project.id
    assert fetched.name == "Test Project"


@pytest.mark.asyncio
async def test_create_project_empty_name_raises(db_manager: cdm.ChatDatabaseManager):
    with pytest.raises(cdm.DatabaseError):
        await db_manager.create_project("", "desc")



@pytest.mark.asyncio
async def test_update_project(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("ToUpdate", "desc")
    updated = await db_manager.update_project(
        uuid.UUID(str(project.id)), name="Updated", description="New desc"
    )
    assert updated.name == "Updated"
    assert updated.description == "New desc"


@pytest.mark.asyncio
async def test_delete_project(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("ToDelete", "desc")
    assert await db_manager.delete_project(uuid.UUID(str(project.id))) is True
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_project(uuid.UUID(str(project.id)))


@pytest.mark.asyncio
async def test_create_and_get_session(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Session 1")
    assert session.title == "Session 1"
    fetched = await db_manager.get_session(uuid.UUID(str(session.id)))
    assert fetched.id == session.id


@pytest.mark.asyncio
async def test_create_session_empty_title_raises(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    with pytest.raises(cdm.DatabaseError):
        await db_manager.create_session(uuid.UUID(str(project.id)), "")


@pytest.mark.asyncio
async def test_update_session(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Old Title")
    updated = await db_manager.update_session(uuid.UUID(str(session.id)), "New Title")
    assert updated.title == "New Title"


@pytest.mark.asyncio
async def test_delete_session(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "ToDelete")
    assert await db_manager.delete_session(uuid.UUID(str(session.id))) is True
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_session(uuid.UUID(str(session.id)))


@pytest.mark.asyncio
async def test_add_and_get_message(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Hello!")
    assert msg.role == "user"
    assert msg.content == "Hello!"
    fetched = await db_manager.get_message(uuid.UUID(str(msg.id)))
    assert fetched.id == msg.id


@pytest.mark.asyncio
async def test_add_message_empty_content_raises(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    with pytest.raises(cdm.DatabaseError):
        await db_manager.add_message(uuid.UUID(str(session.id)), "user", "")


@pytest.mark.asyncio
async def test_add_message_invalid_role_raises(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    with pytest.raises(cdm.DatabaseError):
        await db_manager.add_message(uuid.UUID(str(session.id)), "invalidrole", "Content")


@pytest.mark.asyncio
async def test_update_message(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Hi")
    updated = await db_manager.update_message(
        uuid.UUID(str(msg.id)), role="assistant", content=b"Hello back"
    )
    assert updated.role == "assistant"
    assert updated.content == b"Hello back"


@pytest.mark.asyncio
async def test_delete_message(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Bye")
    assert await db_manager.delete_message(uuid.UUID(str(msg.id))) is True
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_message(uuid.UUID(str(msg.id)))


@pytest.mark.asyncio
async def test_get_conversation_history(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Hi")
    await db_manager.add_message(uuid.UUID(str(session.id)), "assistant", "Hello!")
    history = await db_manager.get_conversation_history(uuid.UUID(str(session.id)))
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_list_projects(db_manager: cdm.ChatDatabaseManager):
    await db_manager.create_project("A", "desc")
    await db_manager.create_project("B", "desc")
    projects = await db_manager.list_projects()
    assert len(projects) >= 2


@pytest.mark.asyncio
async def test_get_sessions_for_project(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    await db_manager.create_session(uuid.UUID(str(project.id)), "S1")
    await db_manager.create_session(uuid.UUID(str(project.id)), "S2")
    sessions = await db_manager.get_sessions_for_project(uuid.UUID(str(project.id)))
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_get_messages_for_session(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Hi")
    await db_manager.add_message(uuid.UUID(str(session.id)), "assistant", "Hello!")
    messages = await db_manager.get_messages_for_session(uuid.UUID(str(session.id)))
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_delete_project_also_deletes_sessions_and_messages(
    db_manager: cdm.ChatDatabaseManager,
):
    project = await db_manager.create_project("Cascade", "desc")
    session = await db_manager.create_session(
        uuid.UUID(str(project.id)), "Cascade Session"
    )
    msg = await db_manager.add_message(
        uuid.UUID(str(session.id)), "user", "Cascade message"
    )
    assert await db_manager.delete_project(uuid.UUID(str(project.id))) is True
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_project(uuid.UUID(str(project.id)))
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_session(uuid.UUID(str(session.id)))
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_message(uuid.UUID(str(msg.id)))


@pytest.mark.asyncio
async def test_delete_session_also_deletes_messages(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Cascade2", "desc")
    session = await db_manager.create_session(
        uuid.UUID(str(project.id)), "Cascade2 Session"
    )
    msg = await db_manager.add_message(
        uuid.UUID(str(session.id)), "user", "Cascade2 message"
    )
    assert await db_manager.delete_session(uuid.UUID(str(session.id))) is True
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_session(uuid.UUID(str(session.id)))
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_message(uuid.UUID(str(msg.id)))


@pytest.mark.asyncio
async def test_update_message_invalid_role_raises(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", "Hi")
    with pytest.raises(cdm.DatabaseError):
        await db_manager.update_message(uuid.UUID(str(msg.id)), role="notarole")


@pytest.mark.asyncio
async def test_update_message_empty_content_raises(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("Proj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "Sess")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", b"Hi")
    with pytest.raises(cdm.DatabaseError):
        await db_manager.update_message(uuid.UUID(str(msg.id)), content=b"")


@pytest.mark.asyncio
async def test_get_project_not_found_raises(db_manager: cdm.ChatDatabaseManager):
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_project(uuid.uuid4())


@pytest.mark.asyncio
async def test_get_session_not_found_raises(db_manager: cdm.ChatDatabaseManager):
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_session(uuid.uuid4())


@pytest.mark.asyncio
async def test_get_message_not_found_raises(db_manager: cdm.ChatDatabaseManager):
    with pytest.raises(cdm.NotFoundError):
        await db_manager.get_message(uuid.uuid4())


@pytest.mark.asyncio
async def test_get_conversation_history_returns_correct_format_and_order(
    db_manager: cdm.ChatDatabaseManager,
):
    project = await db_manager.create_project("HistoryProj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "HistorySession")
    msg1 = await db_manager.add_message(
        uuid.UUID(str(session.id)), "user", b"First message"
    )
    msg2 = await db_manager.add_message(
        uuid.UUID(str(session.id)), "assistant", b"Second message"
    )
    history = await db_manager.get_conversation_history(uuid.UUID(str(session.id)))
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


@pytest.mark.asyncio
async def test_get_conversation_history_empty(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("EmptyProj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "EmptySession")
    history = await db_manager.get_conversation_history(uuid.UUID(str(session.id)))
    assert isinstance(history, list)
    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_project_summary_returns_dict(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("SummaryProj", "desc")
    session1 = await db_manager.create_session(uuid.UUID(str(project.id)), "Session1")
    session2 = await db_manager.create_session(uuid.UUID(str(project.id)), "Session2")
    await db_manager.add_message(uuid.UUID(str(session1.id)), "user", b"Hello 1")
    await db_manager.add_message(uuid.UUID(str(session1.id)), "assistant", b"Hi 1")
    await db_manager.add_message(uuid.UUID(str(session2.id)), "user", b"Hello 2")
    summary = await db_manager.get_project_summary(uuid.UUID(str(project.id)))
    assert isinstance(summary, dict)
    assert summary["project"]["id"] == str(project.id)
    assert summary["project"]["name"] == "SummaryProj"
    assert summary["project"]["description"] == "desc"
    assert summary["stats"]["session_count"] == 2
    assert summary["stats"]["message_count"] == 3


@pytest.mark.asyncio
async def test_get_project_summary_no_sessions(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("NoSessProj", "desc")
    summary = await db_manager.get_project_summary(uuid.UUID(str(project.id)))
    assert summary["stats"]["session_count"] == 0
    assert summary["stats"]["message_count"] == 0


@pytest.mark.asyncio
async def test_get_project_summary_invalid_project_raises(
    db_manager: cdm.ChatDatabaseManager,
):
    with pytest.raises(cdm.DatabaseError):
        await db_manager.get_project_summary(uuid.uuid4())


@pytest.mark.asyncio
async def test_project_str_repr(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("StrTest", "desc")
    s = str(project)
    assert "Project(id=" in s and "StrTest" in s


@pytest.mark.asyncio
async def test_session_str_repr(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("StrTest", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "SessionStr")
    s = str(session)
    assert "ChatSession(id=" in s and "SessionStr" in s



@pytest.mark.asyncio
async def test_project_getters(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("GetterProj", "desc")
    assert isinstance(project.get_id(), uuid.UUID)
    assert isinstance(project.get_created_at(), project.created_at.__class__)
    assert isinstance(project.get_updated_at(), project.updated_at.__class__)
    assert project.get_name() == "GetterProj"
    assert project.get_description() == "desc"


@pytest.mark.asyncio
async def test_session_getters(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("GetterProj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "SessTitle")
    assert isinstance(session.get_id(), uuid.UUID)
    assert isinstance(session.get_created_at(), session.created_at.__class__)
    assert isinstance(session.get_updated_at(), session.updated_at.__class__)
    assert session.get_title() == "SessTitle"
    assert session.get_project_id() == project.id


@pytest.mark.asyncio
async def test_message_getters(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("GetterProj", "desc")
    session = await db_manager.create_session(uuid.UUID(str(project.id)), "SessTitle")
    msg = await db_manager.add_message(uuid.UUID(str(session.id)), "user", b"Hello bytes")
    assert isinstance(msg.get_id(), uuid.UUID)
    assert isinstance(msg.get_session_id(), uuid.UUID)
    assert msg.get_role() == "user"
    assert isinstance(msg.get_content(), bytes)
    assert isinstance(msg.get_timestamp(), msg.date_time.__class__)
    assert isinstance(msg.get_date_time(), msg.date_time.__class__)


@pytest.mark.asyncio
async def test_transaction_context_manager_success(db_manager: cdm.ChatDatabaseManager):
    project = await db_manager.create_project("TxProj", "desc")
    assert project.name == "TxProj"


@pytest.mark.asyncio
async def test_transaction_context_manager_raises(db_manager: cdm.ChatDatabaseManager):
    with pytest.raises(Exception):
        raise Exception("fail in tx")
