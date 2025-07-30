import random
import time
from pydantic import SecretStr
import pytest
from fastapi import WebSocketDisconnect, status
import ulid
from typing import Optional
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    DownloadTaskStatus,
    SearchModelsManifestRequest,
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest
from starlette.testclient import WebSocketDenialResponse

DOWNLOAD_TIMEOUT = 180
MAX_MODELS_TO_DOWNLOAD = 2
TEST_PASSWORD = "Saturate-Heave8-Unfasten-Squealing"
SEARCH_LIMIT = 100


@pytest.fixture
def unlocked_client(integration_client):
    orchestrator = integration_client.app.state.orchestrator
    assert orchestrator.state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(password=SecretStr(TEST_PASSWORD))

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Vault initialized and unlocked."
    assert orchestrator.state == AppState.UNLOCKED
    return integration_client


def _monitor_download_progress(client, task_id: str, timeout: int = DOWNLOAD_TIMEOUT):
    orchestrator = client.app.state.orchestrator
    final_status = None
    
    try:
        print(f"Attempting WebSocket connection for task_id: {task_id}")
        print(f"Orchestrator state: {orchestrator.state}")
        
        with client.websocket_connect(
            f"/api/v1/models_manager/download_progress/{task_id}",
            headers={"Host": "test"},
        ) as websocket:
            timeout_time = time.time() + timeout
            while time.time() < timeout_time:
                message = websocket.receive_json()
                final_status = message
                
                if str(message.get("status")) in [
                    DownloadTaskStatus.COMPLETED.value, 
                    DownloadTaskStatus.FAILED.value
                ]:
                    break
                    
    except WebSocketDisconnect as e:
        pytest.fail(
            "WebSocket connection was unexpectedly disconnected. "
            f"Disconnection details: {e}"
        )
    except WebSocketDenialResponse as e:
        task_check = client.get(f"/api/v1/models_manager/download_status/{task_id}")
        print(
            f"Task status check: {task_check.status_code}, "
            f"{task_check.json() if task_check.status_code == 200 else task_check.text}"
        )
        pytest.fail(
            "WebSocket connection was denied by the server. "
            f"Denial details: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"WebSocket connection failed with an unexpected error: "
            f"{type(e).__name__} - {e}"
        )
        
    return final_status


def _validate_model_structure(model: dict):
    required_fields = ["repo_id", "filename", "local_path", "file_size", "organization"]
    
    for field in required_fields:
        assert field in model, f"Missing field: {field}"
        assert model[field] is not None, f"Field {field} should not be None"
    
    assert model["file_size"] >= 0, "File size should be non-negative"


def test_search_models(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="llama", limit=10, filters_tags=["llama"]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Models retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) <= 10, "Expected at most 10 models in the response."

    for model in data["models"]:
        _validate_model_structure(model)


def test_search_models_no_results(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="nonexistentmodel", limit=10, filters_tags=["nonexistent"]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND, f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."


def test_model_download_and_progress_flow(unlocked_client):
    # Search for models
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Models retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) <= SEARCH_LIMIT

    # Select smallest model for faster testing
    model_to_download = min(data["models"], key=lambda x: x["file_size"])
    _validate_model_structure(model_to_download)

    # Initiate download
    download_request = DownloadModelRequest(**model_to_download)
    orchestrator = unlocked_client.app.state.orchestrator

    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_202_ACCEPTED, f"Expected 202 Accepted, got {response.text}"
    download_data = response.json()
    task_id = download_data["task_id"]
    
    assert task_id is not None, "Task ID should not be None."
    assert download_data["status"] in [
        DownloadTaskStatus.PENDING.value,
        DownloadTaskStatus.RUNNING.value,
        DownloadTaskStatus.COMPLETED.value,
    ], f"Unexpected initial status: {download_data['status']}"

    # Monitor download progress
    final_status = _monitor_download_progress(unlocked_client, task_id)

    # Verify download completion
    expected_file_path = (
        orchestrator.directories.data
        / "models"
        / model_to_download["repo_id"]
        / model_to_download["filename"]
    )

    assert expected_file_path.exists(), f"Downloaded file not found at {expected_file_path}"
    assert expected_file_path.is_file()

    assert final_status is not None, "Test timed out waiting for download to complete."
    assert final_status["percentage"] == 1.0
    assert str(final_status["status"]) == DownloadTaskStatus.COMPLETED.value


def test_model_download_not_found(unlocked_client):
    non_existent_task_id = str(ulid.ULID())

    response = unlocked_client.get(
        f"/api/v1/models_manager/download_status/{non_existent_task_id}"
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND, f"Expected 404 Not Found, got {response.text}"
    assert response.json()['detail'] == "No download task found with the provided ID."


def test_cancel_download(unlocked_client):
    # Search and select model
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert isinstance(data["models"], list)

    model_to_download = min(data["models"], key=lambda x: x["file_size"])
    download_request = DownloadModelRequest(**model_to_download)

    orchestrator = unlocked_client.app.state.orchestrator
    assert orchestrator.state == AppState.UNLOCKED, "Orchestrator should be in UNLOCKED state."

    # Start download
    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_202_ACCEPTED, f"Expected 202 Accepted, got {response.text}"
    download_data = response.json()
    task_id = download_data["task_id"]
    
    # Cancel download
    cancel_response = unlocked_client.post(f"/api/v1/models_manager/cancel_download/{task_id}")
    
    assert cancel_response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {cancel_response.text}"
    cancel_data = cancel_response.json()

    assert str(cancel_data["status"]) == DownloadTaskStatus.CANCELLED.value
    assert cancel_data["message"] == "Download task has been cancelled."
    assert cancel_data["task_id"] == task_id


@pytest.fixture
def unlocked_client_with_filled_manifest(unlocked_client):
    # Search for models
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=["llama"]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert isinstance(data["models"], list)
    
    # Download smallest models for testing
    sorted_models = sorted(data["models"], key=lambda x: x["file_size"])
    nb_models_to_download = min(MAX_MODELS_TO_DOWNLOAD, len(sorted_models))

    for model in sorted_models[:nb_models_to_download]:
        _validate_model_structure(model)
        
        download_request = DownloadModelRequest(**model)
        response = unlocked_client.post(
            "/api/v1/models_manager/download_model",
            json=download_request.model_dump(mode="json"),
        )
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        download_data = response.json()
        task_id = download_data["task_id"]
        assert task_id is not None, "Task ID should not be None."
        
        # Monitor download completion
        _monitor_download_progress(unlocked_client, task_id)
            
    return unlocked_client


def _test_manifest_search(client, repo_id: str, filename: str, expected_count: int = MAX_MODELS_TO_DOWNLOAD):
    search_model_request = SearchModelsManifestRequest(
        repo_id=repo_id, filename=filename
    )
    response = client.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Models manifest retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) == expected_count

    for model in data["models"]:
        _validate_model_structure(model)


def test_get_model_info_manifest(unlocked_client_with_filled_manifest):
    _test_manifest_search(unlocked_client_with_filled_manifest, "llama", "llama")


def test_get_model_info_manifest_partial(unlocked_client_with_filled_manifest):
    _test_manifest_search(unlocked_client_with_filled_manifest, "ll", "ma")


def test_get_model_info_manifest_case_insensitive(unlocked_client_with_filled_manifest):
    _test_manifest_search(unlocked_client_with_filled_manifest, "LL", "MA")


def test_get_model_info_manifest_no_results(unlocked_client_with_filled_manifest):
    search_model_request = SearchModelsManifestRequest(
        repo_id="nonexistentmodel", filename="nonexistent"
    )
    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND, f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."