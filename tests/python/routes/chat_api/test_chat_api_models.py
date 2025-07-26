import datetime
import uuid
from pydantic import ValidationError
import pytest

from ataraxai.routes.chat_route.chat_api_models import (
    CreateProjectRequestAPI,
    DeleteProjectRequestAPI,
    CreateSessionRequestAPI,
    ChatMessageRequestAPI,
    MessageResponseAPI,
    ProjectResponseAPI,
    DeleteProjectResponseAPI,
    SessionResponseAPI,
    DeleteSessionResponseAPI,
)

def test_create_project_request_fields():
    req = CreateProjectRequestAPI(name="Test Project", description="A test project.")
    assert req.name == "Test Project"
    assert req.description == "A test project."

def test_delete_project_request_fields():
    pid = uuid.uuid4()
    req = DeleteProjectRequestAPI(project_id=pid)
    assert req.project_id == pid

def test_create_session_request_fields():
    pid = uuid.uuid4()
    req = CreateSessionRequestAPI(project_id=pid, title="Session Title")
    assert req.project_id == pid
    assert req.title == "Session Title"

def test_chat_message_request_fields():
    req = ChatMessageRequestAPI(user_query="Hello AI")
    assert req.user_query == "Hello AI"

def test_message_response_fields():
    sid = uuid.uuid4()
    mid = uuid.uuid4()
    resp = MessageResponseAPI(id=mid, role="user", content="Hi!", date_time=datetime.datetime.now(), session_id=sid)
    assert resp.content == "Hi!"
    assert resp.id == mid

def test_project_response_fields():
    pid = uuid.uuid4()
    resp = ProjectResponseAPI(project_id=pid, name="Proj", description="Desc")
    assert resp.project_id == pid
    assert resp.name == "Proj"
    assert resp.description == "Desc"

def test_delete_project_response_fields():
    pid = uuid.uuid4()
    resp = DeleteProjectResponseAPI(project_id=pid, name="Proj", description="Desc", status="deleted")
    assert resp.project_id == pid
    assert resp.name == "Proj"
    assert resp.description == "Desc"
    assert resp.status == "deleted"

def test_session_response_fields():
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    resp = SessionResponseAPI(session_id=sid, title="Title", project_id=pid)
    assert resp.session_id == sid
    assert resp.title == "Title"
    assert resp.project_id == pid

def test_delete_session_response_fields():
    sid = uuid.uuid4()
    pid = uuid.uuid4()
    resp = DeleteSessionResponseAPI(session_id=sid, title="Title", project_id=pid, status="deleted")
    assert resp.session_id == sid
    assert resp.title == "Title"
    assert resp.project_id == pid
    assert resp.status == "deleted"

def test_create_project_request_missing_fields():
    with pytest.raises(ValidationError):
        CreateProjectRequestAPI(name="OnlyName")

def test_delete_project_request_invalid_uuid():
    with pytest.raises(ValueError):
        DeleteProjectRequestAPI(project_id="not-a-uuid")