import asyncio
from typing import Generator
from typing import Any, Generator
import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient
from helpers import (
    MAX_MODELS_TO_DOWNLOAD,
    SEARCH_LIMIT,
    clean_downloaded_models,
    monitor_download_progress,
    validate_model_structure,
)
from pydantic import SecretStr

from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.models_manager_route.models_manager_api_models import (
    DownloadModelRequest,
    SearchModelsRequest,
)
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest

# @pytest.fixture(scope="module")
# async def module_unlocked_client(module_integration_client):
#     orchestrator : AtaraxAIOrchestrator = module_integration_client.app.state.orchestrator
#     state = await orchestrator.get_state()
#     assert state == AppState.FIRST_LAUNCH

#     password_request = VaultPasswordRequest(
#         password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
#     )

#     response = module_integration_client.post(
#         "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
#     )

#     assert response.status_code == status.HTTP_200_OK, f"Unexpected status code: {response.text}"
#     data = response.json()
#     assert data["status"] == Status.SUCCESS
#     assert data["message"] == "Vault initialized and unlocked."
#     state = await orchestrator.get_state()
#     assert state == AppState.UNLOCKED
#     return module_integration_client


# @pytest.fixture(scope="module")
@pytest.fixture(scope="module")
def module_unlocked_client(
    module_integration_client: TestClient, event_loop: asyncio.AbstractEventLoop
) -> Generator[TestClient, None, None]:
    """
    A module-scoped fixture that ensures the application is in an UNLOCKED state.
    It intelligently checks the current state and performs the necessary actions
    (initialize or unlock) only once for the entire test module.
    """
    orchestrator: AtaraxAIOrchestrator = module_integration_client.app.state.orchestrator  # type: ignore

    async def setup_unlock():
        state = await orchestrator.get_state()  # type: ignore
        password = SecretStr("test-password-123")
        password_request = VaultPasswordRequest(password=password)

        if state == AppState.FIRST_LAUNCH:
            # If it's the first launch, we need to initialize the vault
            init_response = module_integration_client.post(
                "/api/v1/vault/initialize",
                json=password_request.model_dump(mode="json"),
            )
            assert (
                init_response.status_code == status.HTTP_200_OK
            ), "Failed to initialize vault in test fixture"

        elif state == AppState.LOCKED:
            # If it's already initialized but locked, we just need to unlock it
            unlock_response = module_integration_client.post(
                "/api/v1/vault/unlock", json=password_request.model_dump(mode="json")
            )
            assert (
                unlock_response.status_code == status.HTTP_200_OK
            ), "Failed to unlock vault in test fixture"

        final_state = await orchestrator.get_state()  # type: ignore
        assert (
            final_state == AppState.UNLOCKED
        ), f"Orchestrator failed to reach UNLOCKED state. Current state: {final_state}"

    event_loop.run_until_complete(setup_unlock())

    yield module_integration_client

    # # --- Teardown: Re-lock the vault at the end of the module ---
    # async def relock_vault():
    #     state = await orchestrator.get_state()  # type: ignore
    #     if state == AppState.UNLOCKED:
    #         await orchestrator.lock()  # type: ignore

    # event_loop.run_until_complete(relock_vault())


@pytest.fixture(scope="module")
def module_unlocked_client_new(
    module_integration_client: TestClient, event_loop: asyncio.AbstractEventLoop
) -> Generator[TestClient, None, None]:
    orchestrator: AtaraxAIOrchestrator = module_integration_client.app.state.orchestrator  # type: ignore

    password = SecretStr("test-password-123")
    password_request = VaultPasswordRequest(password=password)

    # 1. Initialize the vault
    init_response = module_integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )
    assert (
        init_response.status_code == 200
    ), "Failed to initialize vault in test fixture"

    # 3. Unlock the vault
    unlock_response = module_integration_client.post(
        "/api/v1/vault/unlock", json=password_request.model_dump(mode="json")
    )
    assert unlock_response.status_code == 200, "Failed to unlock vault in test fixture"

    yield module_integration_client

    async def relock_vault():
        state = await orchestrator.get_state()
        if state == AppState.UNLOCKED:
            await orchestrator.lock()

    event_loop.run_until_complete(relock_vault())


@pytest_asyncio.fixture(scope="module")
async def module_unlocked_client_with_filled_manifest(module_unlocked_client : TestClient) :
    search_model_request = SearchModelsRequest(query="llama", limit=SEARCH_LIMIT)
    response = module_unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
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

        await monitor_download_progress(module_unlocked_client, task_id)

    orchestrator: AtaraxAIOrchestrator = module_unlocked_client.app.state.orchestrator  # type: ignore
    if orchestrator.services.background_task_manager:
        orchestrator.services.background_task_manager.wait_for_all_tasks(timeout=300)

    yield module_unlocked_client

    clean_downloaded_models(module_unlocked_client)


@pytest_asyncio.fixture(scope="function")
async def unlocked_client(integration_client):
    orchestrator: AtaraxAIOrchestrator = integration_client.app.state.orchestrator
    state = await orchestrator.get_state()
    assert state == AppState.FIRST_LAUNCH

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Unexpected status code: {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert data["message"] == "Vault initialized and unlocked."
    state = await orchestrator.get_state()
    assert state == AppState.UNLOCKED
    return integration_client


@pytest_asyncio.fixture(scope="function")
async def unlocked_client_with_filled_manifest(unlocked_client : TestClient):
    search_model_request = SearchModelsRequest(query="llama", limit=SEARCH_LIMIT)
    # unlocked_client = await unlocked_client
    response = unlocked_client.post(
        "/api/v1/models_manager/search_models",
        json=search_model_request.model_dump(mode="json"),
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
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

        await monitor_download_progress(unlocked_client, task_id)

    yield unlocked_client

    clean_downloaded_models(unlocked_client)
