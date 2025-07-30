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
from helpers import (
    monitor_download_progress,
    clean_downloaded_models,
    validate_model_structure,
    SEARCH_LIMIT,
    MAX_MODELS_TO_DOWNLOAD
)


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

    for model in data["models"]:
        validate_model_structure(model)


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
    # Search for models
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
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
    assert len(data["models"]) <= SEARCH_LIMIT

    # Select smallest model for faster testing
    model_to_download = min(data["models"], key=lambda x: x["file_size"])
    validate_model_structure(model_to_download)

    # Initiate download
    download_request = DownloadModelRequest(**model_to_download)
    orchestrator = unlocked_client.app.state.orchestrator

    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    download_data = response.json()
    task_id = download_data["task_id"]

    assert task_id is not None, "Task ID should not be None."
    assert download_data["status"] in [
        DownloadTaskStatus.PENDING.value,
        DownloadTaskStatus.RUNNING.value,
        DownloadTaskStatus.COMPLETED.value,
    ], f"Unexpected initial status: {download_data['status']}"

    # Monitor download progress
    final_status = monitor_download_progress(unlocked_client, task_id)

    # Verify download completion
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

    clean_downloaded_models(unlocked_client)


def test_model_download_not_found(unlocked_client):
    non_existent_task_id = str(ulid.ULID())

    response = unlocked_client.get(
        f"/api/v1/models_manager/download_status/{non_existent_task_id}"
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    assert response.json()["detail"] == "No download task found with the provided ID."


def test_cancel_download(unlocked_client):
    # Search and select model
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
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

    model_to_download = min(data["models"], key=lambda x: x["file_size"])
    download_request = DownloadModelRequest(**model_to_download)

    orchestrator = unlocked_client.app.state.orchestrator
    assert (
        orchestrator.state == AppState.UNLOCKED
    ), "Orchestrator should be in UNLOCKED state."

    # Start download
    response = unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    download_data = response.json()
    task_id = download_data["task_id"]

    # Cancel download
    cancel_response = unlocked_client.post(
        f"/api/v1/models_manager/cancel_download/{task_id}"
    )

    assert (
        cancel_response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {cancel_response.text}"
    cancel_data = cancel_response.json()

    assert str(cancel_data["status"]) == DownloadTaskStatus.CANCELLED.value
    assert cancel_data["message"] == "Download task has been cancelled."
    assert cancel_data["task_id"] == task_id

    clean_downloaded_models(unlocked_client)





def _test_manifest_search(
    modified_client,
    repo_id: str,
    filename: str,
    expected_count: int = MAX_MODELS_TO_DOWNLOAD,
):
    search_model_request = SearchModelsManifestRequest(
        repo_id=repo_id, filename=filename
    )
    response = modified_client.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {response.text}"
    data = response.json()

    assert data["status"] == Status.SUCCESS, f"Expected success status, got {data}"
    assert data["message"] == "Model information retrieved successfully."
    assert isinstance(data["models"], list)
    assert len(data["models"]) == expected_count

    for model in data["models"]:
        validate_model_structure(model)


def test_get_model_info_manifest(unlocked_client_with_filled_manifest):
    _test_manifest_search(
        unlocked_client_with_filled_manifest, repo_id="llama", filename="llama"
    )
    clean_downloaded_models(unlocked_client_with_filled_manifest)


def test_get_model_info_manifest_partial(unlocked_client_with_filled_manifest):
    _test_manifest_search(
        unlocked_client_with_filled_manifest, repo_id="ll", filename="ma"
    )
    clean_downloaded_models(unlocked_client_with_filled_manifest)


def test_get_model_info_manifest_case_insensitive(unlocked_client_with_filled_manifest):
    _test_manifest_search(
        unlocked_client_with_filled_manifest, repo_id="LL", filename="MA"
    )
    clean_downloaded_models(unlocked_client_with_filled_manifest)


def test_get_model_info_manifest_no_results(unlocked_client_with_filled_manifest):
    search_model_request = SearchModelsManifestRequest(
        repo_id="nonexistentmodel", filename="nonexistent"
    )
    response = unlocked_client_with_filled_manifest.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."
    clean_downloaded_models(unlocked_client_with_filled_manifest)
