from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest
from fastapi import status

from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest
from helpers import (
    monitor_download_progress,
    clean_downloaded_models,
    validate_model_structure,
    SEARCH_LIMIT,
    MAX_MODELS_TO_DOWNLOAD,
)




@pytest.fixture(scope="module")
async def module_unlocked_client(module_integration_client):
    orchestrator : AtaraxAIOrchestrator = module_integration_client.app.state.orchestrator
    state = await orchestrator.get_state()
    assert state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = module_integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Vault initialized and unlocked."
    state = await orchestrator.get_state()
    assert state == AppState.UNLOCKED
    return module_integration_client



@pytest.fixture(scope="module")
def module_unlocked_client_with_filled_manifest(module_unlocked_client):
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT, filters_tags=["llama"]
    )
    response = module_unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert isinstance(data["models"], list)

    sorted_models = sorted(data["models"], key=lambda x: x["file_size"])
    nb_models_to_download = min(MAX_MODELS_TO_DOWNLOAD, len(sorted_models))

    for model in sorted_models[:nb_models_to_download]:
        validate_model_structure(model)

        download_request = DownloadModelRequest(**model)
        response = module_unlocked_client.post(
            "/api/v1/models_manager/download_model",
            json=download_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        download_data = response.json()
        task_id = download_data["task_id"]
        assert task_id is not None, "Task ID should not be None."

        monitor_download_progress(module_unlocked_client, task_id)

    yield module_unlocked_client

    clean_downloaded_models(module_unlocked_client)


@pytest.fixture(scope="function")
async def unlocked_client(integration_client):
    orchestrator : AtaraxAIOrchestrator = integration_client.app.state.orchestrator
    state = await orchestrator.get_state()
    assert state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK, f"Unexpected status code: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert data["message"] == "Vault initialized and unlocked."
    state = await orchestrator.get_state()
    assert state == AppState.UNLOCKED
    return integration_client



@pytest.fixture(scope="function")
def unlocked_client_with_filled_manifest(unlocked_client):
    search_model_request = SearchModelsRequest(
        query="llama", limit=SEARCH_LIMIT
    )
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert isinstance(data["models"], list)

    sorted_models = sorted(data["models"], key=lambda x: x["file_size"])
    nb_models_to_download = min(MAX_MODELS_TO_DOWNLOAD, len(sorted_models))

    for model in sorted_models[:nb_models_to_download]:
        validate_model_structure(model)

        download_request = DownloadModelRequest(**model)
        response = unlocked_client.post(
            "/api/v1/models_manager/download_model",
            json=download_request.model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        download_data = response.json()
        task_id = download_data["task_id"]
        assert task_id is not None, "Task ID should not be None."

        monitor_download_progress(unlocked_client, task_id)

    yield unlocked_client

    clean_downloaded_models(unlocked_client)
