from unittest import mock
from fastapi import status
from pydantic import SecretStr
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import (
        VaultPasswordRequest,
        ConfirmationPhaseRequest,
    )

def test_reinitialize_vault_success(integration_client : mock.MagicMock):
    orchestrator = integration_client.app.state.orchestrator
    password_request = VaultPasswordRequest(password=SecretStr("Saturate-Heave8-Unfasten-Squealing"))
    integration_client.post("/api/v1/vault/initialize", json=password_request.model_dump(mode="json"))
    confirmation_request = ConfirmationPhaseRequest(confirmation_phrase="CONFIRM-RESET")
    with mock.patch.object(orchestrator, "reinitialize_vault", return_value=True):
        response = integration_client.post(
            "/api/v1/vault/reinitialize", json=confirmation_request.model_dump(mode="json")
        )
    assert response.status_code == status.HTTP_200_OK, f"Expected status code 200, got {response.text}"
    data = response.json()
    assert data["status"] == Status.SUCCESS
    assert "Vault reinitialized" in data["message"]

def test_reinitialize_vault_failure(integration_client : mock.MagicMock):
    orchestrator = integration_client.app.state.orchestrator
    password_request = VaultPasswordRequest(password=SecretStr("Saturate-Heave8-Unfasten-Squealing"))
    integration_client.post("/api/v1/vault/initialize", json=password_request.model_dump(mode="json"))
    confirmation_request = ConfirmationPhaseRequest(confirmation_phrase="CONFIRM-RESET")
    with mock.patch.object(orchestrator, "reinitialize_vault", return_value=False):
        response = integration_client.post(
            "/api/v1/vault/reinitialize", json=confirmation_request.model_dump(mode="json")
        )
    assert response.status_code == status.HTTP_200_OK, f"Expected status code 200, got {response.text}"
    data = response.json()
    assert data["status"] == Status.ERROR
    assert "Failed to reinitialize vault" in data["message"]
