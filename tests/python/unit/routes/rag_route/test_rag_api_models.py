import pytest
from pydantic import ValidationError
from ataraxai.routes.status import Status


from ataraxai.routes.rag_route.rag_api_models import (
    CheckManifestResponse,
    DirectoriesToScanRequest,
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
    RebuildIndexResponse,
    DirectoriesAdditionResponse,
    DirectoriesRemovalResponse,
)

def test_check_manifest_response_valid():
    resp = CheckManifestResponse(status=Status.SUCCESS, message="Manifest checked.")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Manifest checked."

def test_check_manifest_response_empty_message():
    with pytest.raises(ValidationError):
        CheckManifestResponse(status=Status.SUCCESS, message="")

def test_directories_to_scan_request_valid():
    req = DirectoriesToScanRequest(directories=["/data", "/tmp"])
    assert req.directories == ["/data", "/tmp"]

def test_directories_to_scan_request_empty_list():
    with pytest.raises(ValidationError):
        DirectoriesToScanRequest(directories=[])

def test_directories_to_scan_request_whitespace_entry():
    with pytest.raises(ValidationError):
        DirectoriesToScanRequest(directories=["/data", "   "])

def test_directories_to_add_request_valid():
    req = DirectoriesToAddRequest(directories=["/foo", "/bar"])
    assert req.directories == ["/foo", "/bar"]

def test_directories_to_remove_request_valid():
    req = DirectoriesToRemoveRequest(directories=["/foo", "/bar"])
    assert req.directories == ["/foo", "/bar"]

def test_rebuild_index_response_valid():
    resp = RebuildIndexResponse(status=Status.SUCCESS, message="Index rebuilt.", result={"count": 1})
    assert resp.status == Status.SUCCESS
    assert resp.message == "Index rebuilt."
    assert resp.result == {"count": 1}

def test_directories_addition_response_valid():
    resp = DirectoriesAdditionResponse(status=Status.SUCCESS, message="Added.", result=["/foo"])
    assert resp.status == Status.SUCCESS
    assert resp.message == "Added."
    assert resp.result == ["/foo"]

def test_directories_removal_response_valid():
    resp = DirectoriesRemovalResponse(status=Status.SUCCESS, message="Removed.", result=["/foo"])
    assert resp.status == Status.SUCCESS
    assert resp.message == "Removed."
    assert resp.result == ["/foo"]
