import pytest
from pydantic import ValidationError
from ataraxai.routes.status import Status

from ataraxai.routes.rag_api_models import (
    CheckManifestResponse,
    DirectoriesToScanRequest,
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
    RebuildIndexResponse,
    ScanAndIndexResponse,
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
    req = DirectoriesToScanRequest(directories=["/data", "/home"])
    assert req.directories == ["/data", "/home"]


def test_directories_to_scan_request_empty_list():
    with pytest.raises(ValidationError):
        DirectoriesToScanRequest(directories=[])


def test_directories_to_scan_request_whitespace_directory():
    with pytest.raises(ValidationError):
        DirectoriesToScanRequest(directories=["   "])


def test_directories_to_add_request_valid():
    req = DirectoriesToAddRequest(directories=["/add1", "/add2"])
    assert req.directories == ["/add1", "/add2"]


def test_directories_to_remove_request_valid():
    req = DirectoriesToRemoveRequest(directories=["/remove1", "/remove2"])
    assert req.directories == ["/remove1", "/remove2"]


def test_rebuild_index_response_valid():
    resp = RebuildIndexResponse(status=Status.SUCCESS, message="Index rebuilt.")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Index rebuilt."


def test_scan_and_index_response_valid():
    resp = ScanAndIndexResponse(status=Status.SUCCESS, message="Scan complete.")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Scan complete."


def test_directories_addition_response_valid():
    resp = DirectoriesAdditionResponse(status=Status.SUCCESS, message="Added directories.")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Added directories."


def test_directories_removal_response_valid():
    resp = DirectoriesRemovalResponse(status=Status.SUCCESS, message="Removed directories.")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Removed directories."