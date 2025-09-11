from unittest import mock

from fastapi import status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import (
    ConfirmationPhaseRequest,
    VaultPasswordRequest,
)


def test_reinitialize_vault_success(integration_client: TestClient):
    orchestrator = integration_client.app.state.orchestrator
    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    # Initialize the vault first
    integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    # Override the specific dependency used by the reinitialize endpoint
    integration_client.app.dependency_overrides[get_unlocked_orchestrator] = (
        lambda: orchestrator
    )

    confirmation_request = ConfirmationPhaseRequest(confirmation_phrase="CONFIRM-RESET")
    with mock.patch.object(orchestrator, "reinitialize_vault", return_value=True):
        response = integration_client.post(
            "/api/v1/vault/reinitialize",
            json=confirmation_request.model_dump(mode="json"),
        )

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected status code 200, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS.value
    assert "Vault reinitialized" in data["message"]


def test_reinitialize_vault_failure(integration_client: TestClient):
    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    confirmation_request = ConfirmationPhaseRequest(confirmation_phrase="CONFIRM-RESET")

    mock_orchestrator = mock.MagicMock()
    mock_orchestrator.reinitialize_vault.return_value = False

    integration_client.app.dependency_overrides[get_unlocked_orchestrator] = (
        lambda: mock_orchestrator
    )

    response = integration_client.post(
        "/api/v1/vault/reinitialize", json=confirmation_request.model_dump(mode="json")
    )

    mock_orchestrator.reinitialize_vault.assert_called_once_with("CONFIRM-RESET")

    assert (
        response.status_code == status.HTTP_200_OK
    ), f"Expected status code 200, got {response.text}"
    data = response.json()
    assert data["status"] == Status.ERROR.value, f"Expected status ERROR, got {data}"
    assert "Failed to reinitialize vault" in data["message"]
