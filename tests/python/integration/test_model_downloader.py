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
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest
from starlette.testclient import WebSocketDenialResponse


@pytest.fixture
def unlocked_client(integration_client):
    """
    Fixture to ensure the vault is unlocked before running tests.
    """
    orchestrator = integration_client.app.state.orchestrator
    assert orchestrator.state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Vault initialized and unlocked."
    assert orchestrator.state == AppState.UNLOCKED
    return integration_client


def test_search_models(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="llama", limit=10, filters_tags=["llama"]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Models retrieved successfully."
    assert isinstance(data["models"], list)

    assert len(data["models"]) <= 10, "Expected at most 10 models in the response."

    if data["models"]:
        for model in data["models"]:
            assert "repo_id" in model
            assert model["repo_id"] is not None
            assert "filename" in model
            assert model["filename"] is not None
            assert "local_path" in model
            assert model["local_path"] is not None
            assert "file_size" in model
            assert model["file_size"] >= 0, "File size should be non-negative."
            assert "organization" in model
            assert model["organization"] is not None, "Organization should not be None."


def test_search_models_no_results(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="nonexistentmodel", limit=10, filters_tags=["nonexistent"]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."
    


def test_model_download_and_progress_flow(unlocked_client):

    search_model_request = SearchModelsRequest(
        query="llama", limit=100, filters_tags=[]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert (
        data["message"] == "Models retrieved successfully."
    ), f"Unexpected message: {data['message']}"
    assert isinstance(data["models"], list)

    assert len(data["models"]) <= 100, "Expected at most 100 models in the response."

    # We select the smallest model to download for testing purposes
    # This is to ensure the download completes quickly in CI environments.
    model_to_download = sorted(data["models"], key=lambda x: x["file_size"])[0]

    assert "organization" in model_to_download
    assert (
        model_to_download["organization"] is not None
    ), "Organization should not be None."
    assert "repo_id" in model_to_download
    assert model_to_download["repo_id"] is not None
    assert "filename" in model_to_download
    assert model_to_download["filename"] is not None
    assert "local_path" in model_to_download
    assert model_to_download["local_path"] is not None
    assert "file_size" in model_to_download

    download_request = DownloadModelRequest(**model_to_download)

    orchestrator = unlocked_client.app.state.orchestrator

    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    data = response.json()
    task_id = data["task_id"]
    assert task_id is not None, "Task ID should not be None."
    assert data["status"] in [
        DownloadTaskStatus.PENDING.value,
        DownloadTaskStatus.RUNNING.value,
        DownloadTaskStatus.COMPLETED.value,
    ], f"Expected status to be PENDING or RUNNING or COMPLETED, got {data['status']} but {DownloadTaskStatus.PENDING.value}"


    final_status = None
    try:
        print(f"Attempting WebSocket connection for task_id: {task_id}")
        print(f"Orchestrator state: {orchestrator.state}")
        # with unlocked_client.websocket_connect(f"ws://test/api/v1/models_manager/download_progress/{task_id}") as websocket:
        with unlocked_client.websocket_connect(
            f"/api/v1/models_manager/download_progress/{task_id}",
            headers={"Host": "test"},
        ) as websocket:
            timeout = time.time() + 180
            while time.time() < timeout:
                message = websocket.receive_json()
                final_status = message
                # print(f"Received message: {message}")
                if str(message.get("status")) in [DownloadTaskStatus.COMPLETED.value, DownloadTaskStatus.FAILED.value] :
                    break
    except WebSocketDisconnect as e:
        pytest.fail(
            "WebSocket connection was unexpectedly disconnected. "
            f"Disconnection details: {e}"
        )
    except WebSocketDenialResponse as e:
        task_check = unlocked_client.get(
            f"/api/v1/models_manager/download_status/{task_id}"
        )
        print(
            f"Task status check: {task_check.status_code}, {task_check.json() if task_check.status_code == 200 else task_check.text}"
        )
        pytest.fail(
            "WebSocket connection was denied by the server. " f"Denial details: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"WebSocket connection failed with an unexpected error: {type(e).__name__} - {e}"
        )
        
    expected_file_path = (
        orchestrator.directories.data
        / "models"
        / model_to_download["repo_id"]
        / model_to_download["filename"]
    )

    assert (
        expected_file_path.exists()
    ), f"Downloaded file not found at {expected_file_path}"
    assert expected_file_path.is_file()

    assert final_status is not None, "Test timed out waiting for download to complete."
    assert final_status["percentage"] == 1.0
    assert str(final_status["status"]) == DownloadTaskStatus.COMPLETED.value
    


def test_model_download_not_found(unlocked_client):
    non_existent_task_id = str(ulid.ULID())

    response = unlocked_client.get(
        f"/api/v1/models_manager/download_status/{non_existent_task_id}"
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    assert response.json()['detail'] == "No download task found with the provided ID."
    
    
def test_cancel_download(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="llama", limit=100, filters_tags=[]
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS
    assert isinstance(data["models"], list)

    model_to_download = sorted(data["models"], key=lambda x: x["file_size"])[0]

    download_request = DownloadModelRequest(**model_to_download)

    orchestrator = unlocked_client.app.state.orchestrator
    assert orchestrator.state == AppState.UNLOCKED, "Orchestrator should be in UNLOCKED state."

    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    data = response.json()
    task_id = data["task_id"]
    
    cancel_response = unlocked_client.post(f"/api/v1/models_manager/cancel_download/{task_id}")
    
    assert cancel_response.status_code == status.HTTP_200_OK, f"Expected 200 OK, got {cancel_response.text}"
    
    cancel_data = cancel_response.json()

    assert str(cancel_data["status"]) == DownloadTaskStatus.CANCELLED.value
    assert cancel_data["message"] == "Download task has been cancelled."
    assert cancel_data["task_id"] == task_id