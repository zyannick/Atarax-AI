from pathlib import Path
from fastapi import status
import ulid
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    DownloadTaskStatus,
    SearchModelsManifestRequest,
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from helpers import (
    SEARCH_LIMIT,
    MAX_MODELS_TO_DOWNLOAD,
    validate_model_structure,
    monitor_download_progress,
    clean_downloaded_models
)


from fastapi.testclient import TestClient

def test_search_models(module_unlocked_client: TestClient):
    search_model_request = SearchModelsRequest(
        query="llama", limit=10, filters_tags=["llama"]
    )
    response = module_unlocked_client.post(
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


def test_search_models_no_results(module_unlocked_client : TestClient):
    search_model_request = SearchModelsRequest(
        query="nonexistentmodel", limit=10, filters_tags=["nonexistent"]
    )
    response = module_unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."


def test_model_download_and_progress_flow(module_unlocked_client : TestClient):
    # Search for models
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
    )
    response = module_unlocked_client.post(
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
    orchestrator = module_unlocked_client.app.state.orchestrator # type: ignore

    response = module_unlocked_client.post(
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
    final_status = monitor_download_progress(module_unlocked_client, task_id)

    # Verify download completion
    expected_file_path : Path = ( # type: ignore
        orchestrator.directories.data # type: ignore
        / "models"
        / model_to_download["repo_id"]
        / model_to_download["filename"]
    )

    assert (
        expected_file_path.exists() # type: ignore
    ), f"Downloaded file not found at {expected_file_path}"
    assert expected_file_path.is_file() # type: ignore

    assert final_status is not None, "Test timed out waiting for download to complete."
    assert final_status["percentage"] == 1.0
    assert str(final_status["status"]) == DownloadTaskStatus.COMPLETED.value

    clean_downloaded_models(module_unlocked_client)


def test_model_download_not_found(module_unlocked_client : TestClient):
    non_existent_task_id = str(ulid.ULID())

    response = module_unlocked_client.get(
        f"/api/v1/models_manager/download_status/{non_existent_task_id}"
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    assert response.json()["detail"] == "No download task found with the provided ID."


def test_cancel_download(module_unlocked_client : TestClient):
    # Search and select model
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=[]
    )
    response = module_unlocked_client.post(
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

    orchestrator = module_unlocked_client.app.state.orchestrator # type: ignore
    assert (
        orchestrator.state == AppState.UNLOCKED # type: ignore
    ), "Orchestrator should be in UNLOCKED state."

    # Start download
    response = module_unlocked_client.post(
        "/api/v1/models_manager/download_model",
        json=download_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_202_ACCEPTED
    ), f"Expected 202 Accepted, got {response.text}"
    download_data = response.json()
    task_id = download_data["task_id"]

    # Cancel download
    cancel_response = module_unlocked_client.post(
        f"/api/v1/models_manager/cancel_download/{task_id}"
    )

    assert (
        cancel_response.status_code == status.HTTP_200_OK
    ), f"Expected 200 OK, got {cancel_response.text}"
    cancel_data = cancel_response.json()

    assert str(cancel_data["status"]) == DownloadTaskStatus.CANCELLED.value
    assert cancel_data["message"] == "Download task has been cancelled."
    assert cancel_data["task_id"] == task_id

    clean_downloaded_models(module_unlocked_client)





def _test_manifest_search(
    modified_client : TestClient,
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


def test_get_model_info_manifest(module_unlocked_client_with_filled_manifest : TestClient):
    _test_manifest_search(
        module_unlocked_client_with_filled_manifest, repo_id="llama", filename="llama"
    )


def test_get_model_info_manifest_partial(module_unlocked_client_with_filled_manifest : TestClient):
    _test_manifest_search(
        module_unlocked_client_with_filled_manifest, repo_id="ll", filename="ma"
    )


def test_get_model_info_manifest_case_insensitive(module_unlocked_client_with_filled_manifest : TestClient):
    _test_manifest_search(
        module_unlocked_client_with_filled_manifest, repo_id="LL", filename="MA"
    )


def test_get_model_info_manifest_no_results(module_unlocked_client_with_filled_manifest : TestClient):
    search_model_request = SearchModelsManifestRequest(
        repo_id="nonexistentmodel", filename="nonexistent"
    )
    response = module_unlocked_client_with_filled_manifest.post(
        "/api/v1/models_manager/get_model_info_manifest",
        json=search_model_request.model_dump(mode="json"),
    )

    assert (
        response.status_code == status.HTTP_404_NOT_FOUND
    ), f"Expected 404 Not Found, got {response.text}"
    data = response.json()
    assert data["detail"] == "No models found matching the search criteria."
