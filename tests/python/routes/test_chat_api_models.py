import uuid
import pytest

from ataraxai.routes.chat_api_models import (
    CreateProjectRequest,
    DeleteProjectRequest,
    CreateSessionRequest,
    ChatMessageRequest,
    MessageResponse,
    ProjectResponse,
    DeleteProjectResponse,
    SessionResponse,
    DeleteSessionResponse,
)

def test_create_project_request_fields():
    req = CreateProjectRequest(name="Test Project", description="A test project.")
    assert req.name == "Test Project"
    assert req.description == "A test project."

def test_delete_project_request_fields():
    pid = uuid.uuid4()
    req = DeleteProjectRequest(project_id=pid)
    assert req.project_id == pid

def test_create_session_request_fields():
    pid = uuid.uuid4()
    req = CreateSessionRequest(project_id=pid, title="Session Title")
    assert req.project_id == pid
    assert req.title == "Session Title"

def test_chat_message_request_fields():
    sid = uuid.uuid4()
    req = ChatMessageRequest(session_id=sid, user_query="Hello AI")
    assert req.session_id == sid
    assert req.user_query == "Hello AI"

def test_message_response_fields():
    sid = uuid.uuid4()
    resp = MessageResponse(assistant_response="Hi!", session_id=sid)
    assert resp.assistant_response == "Hi!"
    assert resp.session_id == sid

def test_project_response_fields():
    pid = uuid.uuid4()
    resp = ProjectResponse(project_id=pid, name="Proj", description="Desc")
    assert resp.project_id == pid
    assert resp.name == "Proj"
    assert resp.description == "Desc"

def test_delete_project_response_fields():
    pid = uuid.uuid4()
    resp = DeleteProjectResponse(project_id=pid, name="Proj", description="Desc", status="deleted")
    assert resp.project_id == pid
    assert resp.name == "Proj"
    assert resp.description == "Desc"
    assert resp.status == "deleted"

def test_session_response_fields():
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    resp = SessionResponse(session_id=sid, title="Title", project_id=pid)
    assert resp.session_id == sid
    assert resp.title == "Title"
    assert resp.project_id == pid

def test_delete_session_response_fields():
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    resp = DeleteSessionResponse(session_id=sid, title="Title", project_id=pid, status="deleted")
    assert resp.session_id == sid
    assert resp.title == "Title"
    assert resp.project_id == pid
    assert resp.status == "deleted"

def test_create_project_request_missing_fields():
    with pytest.raises(TypeError):
        CreateProjectRequest(name="OnlyName")

def test_delete_project_request_invalid_uuid():
    with pytest.raises(ValueError):
        DeleteProjectRequest(project_id="not-a-uuid")