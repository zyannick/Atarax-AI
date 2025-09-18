from datetime import datetime
import uuid
import pytest
from unittest import mock
from ataraxai.praxis.utils.chat_manager import ChatManager


@pytest.mark.asyncio
async def test_create_project_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)

    project_data = {"id": uuid.uuid4(), "name": "Test Project", "description": "Desc"}
    db_manager.create_project.return_value = project_data

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_string") as validate_string, \
         mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", return_value="validated_project") as model_validate:
        result = await chat_manager.create_project("Test Project", "Desc")
        validate_string.assert_called_once_with("Test Project", "Project name")
        db_manager.create_project.assert_awaited_once_with(name="Test Project", description="Desc")
        logger.info.assert_called_once()
        assert result == "validated_project"

@pytest.mark.asyncio
async def test_create_project_failure():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    db_manager.create_project.side_effect = Exception("DB error")

    with pytest.raises(Exception):
        await chat_manager.create_project("Test Project", "Desc")
    logger.error.assert_called_once()

@pytest.mark.asyncio
async def test_get_project_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    project_id = uuid.uuid4()
    db_manager.get_project.return_value = {"id": project_id}

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid, \
         mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", return_value="validated_project") as model_validate:
        result = await chat_manager.get_project(project_id)
        validate_uuid.assert_called_once_with(project_id, "Project ID")
        db_manager.get_project.assert_awaited_once_with(project_id)
        assert result == "validated_project"

@pytest.mark.asyncio
async def test_list_projects_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    db_manager.list_projects.return_value = [{"id": uuid.uuid4()}, {"id": uuid.uuid4()}]

    with mock.patch("ataraxai.praxis.modules.chat.chat_models.ProjectResponse.model_validate", side_effect=lambda x: x):
        result = await chat_manager.list_projects()
        db_manager.list_projects.assert_awaited_once()
        assert isinstance(result, list)
        assert len(result) == 2

@pytest.mark.asyncio
async def test_delete_project_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    project_id = uuid.uuid4()
    db_manager.delete_project.return_value = True

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid:
        result = await chat_manager.delete_project(project_id)
        validate_uuid.assert_called_once_with(project_id, "Project ID")
        db_manager.delete_project.assert_awaited_once_with(project_id)
        logger.info.assert_called_once()
        assert result is True

@pytest.mark.asyncio
async def test_create_session_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    project_id = uuid.uuid4()
    session_data = {"id": uuid.uuid4(), "title": "Session"}
    db_manager.create_session.return_value = session_data

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid, \
         mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_string") as validate_string, \
         mock.patch("ataraxai.praxis.modules.chat.chat_models.ChatSessionResponse.model_validate", return_value="validated_session") as model_validate:
        result = await chat_manager.create_session(project_id, "Session")
        validate_uuid.assert_called_once_with(project_id, "Project ID")
        validate_string.assert_called_once_with("Session", "Session title")
        db_manager.create_session.assert_awaited_once_with(project_id=project_id, title="Session")
        logger.info.assert_called_once()
        assert result == "validated_session"

@pytest.mark.asyncio
async def test_get_session_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    session_id = uuid.uuid4()

    mock_msg = mock.Mock()
    mock_msg.get_id.return_value = uuid.uuid4()
    mock_msg.get_role.return_value = "user"
    mock_msg.get_date_time.return_value = datetime.now()
    mock_msg.content = b"encrypted"
    db_session = mock.Mock()
    db_session.get_id.return_value = session_id
    db_session.get_project_id.return_value = uuid.uuid4()
    db_session.get_title.return_value = "Session"
    db_session.get_created_at.return_value = datetime.now()
    db_session.get_updated_at.return_value = datetime.now()
    db_session.messages = [mock_msg]

    db_manager.get_session.return_value = db_session
    vault_manager.decrypt.return_value = b"decrypted"

    result = await chat_manager.get_session(session_id)
    db_manager.get_session.assert_awaited_once_with(session_id)
    vault_manager.decrypt.assert_called_once_with(b"encrypted")
    assert result.title == "Session"
    assert result.messages[0].content == "decrypted"

@pytest.mark.asyncio
async def test_list_sessions_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    project_id = uuid.uuid4()
    db_manager.get_sessions_for_project.return_value = [{"id": uuid.uuid4()}, {"id": uuid.uuid4()}]

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid, \
         mock.patch("ataraxai.praxis.modules.chat.chat_models.ChatSessionResponse.model_validate", side_effect=lambda x: x):
        result = await chat_manager.list_sessions(project_id)
        validate_uuid.assert_called_once_with(project_id, "Project ID")
        db_manager.get_sessions_for_project.assert_awaited_once_with(project_id)
        assert isinstance(result, list)
        assert len(result) == 2

@pytest.mark.asyncio
async def test_delete_session_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    session_id = uuid.uuid4()
    db_manager.delete_session.return_value = True

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid:
        result = await chat_manager.delete_session(session_id)
        validate_uuid.assert_called_once_with(session_id, "Session ID")
        db_manager.delete_session.assert_awaited_once_with(session_id)
        logger.info.assert_called_once()
        assert result is True

@pytest.mark.asyncio
async def test_add_message_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    session_id = uuid.uuid4()
    role = "user"
    content = "hello"
    encrypted_content = b"encrypted"
    vault_manager.encrypt.return_value = encrypted_content

    db_message = mock.Mock()
    db_message.id = uuid.uuid4()
    db_message.session.id = session_id
    db_message.role = role
    db_message.get_date_time.return_value = datetime.now()
    db_manager.add_message.return_value = db_message

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid, \
         mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_string") as validate_string:
        result = await chat_manager.add_message(session_id, role, content)
        vault_manager.encrypt.assert_called_once_with(content.encode("utf-8"))
        db_manager.add_message.assert_awaited_once_with(session_id=session_id, role=role, encrypted_content=encrypted_content)
        validate_uuid.assert_called_once_with(session_id, "Session ID")
        validate_string.assert_called_once_with(content, "Message content")
        assert result.content == content
        assert result.role == role

@pytest.mark.asyncio
async def test_get_message_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    message_id = uuid.uuid4()

    db_message = mock.Mock()
    db_message.id = message_id
    db_message.session.id = uuid.uuid4()
    db_message.role = "user"
    db_message.content = b"encrypted"
    db_message.get_date_time.return_value = datetime.now()
    db_manager.get_message.return_value = db_message
    vault_manager.decrypt.return_value = b"decrypted"

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid:
        result = await chat_manager.get_message(message_id)
        validate_uuid.assert_called_once_with(message_id, "Message ID")
        db_manager.get_message.assert_awaited_once_with(message_id)
        vault_manager.decrypt.assert_called_once_with(b"encrypted")
        assert result.content == "decrypted"

@pytest.mark.asyncio
async def test_get_messages_for_session_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    session_id = uuid.uuid4()

    mock_msg = mock.Mock()
    mock_msg.get_id.return_value = uuid.uuid4()
    mock_msg.get_session_id.return_value = session_id
    mock_msg.get_role.return_value = "user"
    mock_msg.get_date_time.return_value = datetime.now()
    mock_msg.content = b"encrypted"
    db_manager.get_messages_for_session.return_value = [mock_msg]
    vault_manager.decrypt.return_value = b"decrypted"

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid:
        result = await chat_manager.get_messages_for_session(session_id)
        validate_uuid.assert_called_once_with(session_id, "Session ID")
        db_manager.get_messages_for_session.assert_awaited_once_with(session_id)
        vault_manager.decrypt.assert_called_once_with(b"encrypted")
        assert result[0].content == "decrypted"

@pytest.mark.asyncio
async def test_delete_message_success():
    db_manager = mock.AsyncMock()
    logger = mock.Mock()
    vault_manager = mock.Mock()
    chat_manager = ChatManager(db_manager, logger, vault_manager)
    message_id = uuid.uuid4()
    db_manager.delete_message.return_value = True

    with mock.patch("ataraxai.praxis.utils.input_validator.InputValidator.validate_uuid") as validate_uuid:
        result = await chat_manager.delete_message(message_id)
        validate_uuid.assert_called_once_with(message_id, "Message ID")
        db_manager.delete_message.assert_awaited_once_with(message_id)
        logger.info.assert_called_once()
        assert result is True
