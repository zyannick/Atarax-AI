import pytest
from pydantic import SecretStr
from fastapi import status
from ataraxai.routes.status import Status
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.vault_route.vault_api_models import VaultPasswordRequest




def test_initialise_vault_success(integration_client):
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

    response = integration_client.get("/v1/status")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "UNLOCKED" in data["message"]


def test_initialise_vault_fails_if_already_initialized(integration_client):
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

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )

    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == Status.ERROR
    assert data["message"] == "Vault is already initialized."


def test_full_lock_and_unlock_flow(integration_client):
    orchestrator = integration_client.app.state.orchestrator
    
    password = "Saturate-Heave8-Unfasten-Squealing"
    password_request = VaultPasswordRequest(password=SecretStr(password))
    integration_client.post("/api/v1/vault/initialize", json=password_request.model_dump(mode="json"))
    assert orchestrator.state == AppState.UNLOCKED

    response = integration_client.post("/api/v1/vault/lock")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS
    assert orchestrator.state == AppState.LOCKED

    response = integration_client.post(
        "/api/v1/vault/unlock", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == Status.SUCCESS
    assert orchestrator.state == AppState.UNLOCKED


def test_unlock_fails_with_wrong_password(integration_client):
    orchestrator = integration_client.app.state.orchestrator
   
    correct_password = "Saturate-Heave8-Unfasten-Squealing"
    password_request = VaultPasswordRequest(password=SecretStr(correct_password))
        
    
    response = integration_client.post("/api/v1/vault/initialize", json=password_request.model_dump(mode="json"))
    assert response.status_code == status.HTTP_200_OK
    assert orchestrator.state == AppState.UNLOCKED
    
    lock_response = integration_client.post("/api/v1/vault/lock")
    assert lock_response.status_code == status.HTTP_200_OK
    assert orchestrator.state == AppState.LOCKED
    
    wrong_password_request = VaultPasswordRequest(password=SecretStr("Wrong-Password-123"))
    response = integration_client.post(
        "/api/v1/vault/unlock", json=wrong_password_request.model_dump(mode="json")
    )
    assert response.status_code ==  status.HTTP_401_UNAUTHORIZED

    correct_retry = integration_client.post(
        "/api/v1/vault/unlock", json=password_request.model_dump(mode="json")
    )
    assert correct_retry.status_code == status.HTTP_200_OK
    assert correct_retry.json()["status"] == Status.SUCCESS
    assert orchestrator.state == AppState.UNLOCKED
    
def test_already_unlocked_vault(integration_client):
    orchestrator = integration_client.app.state.orchestrator
    
    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )
    
    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    assert orchestrator.state == AppState.UNLOCKED

    response = integration_client.post(
        "/api/v1/vault/unlock", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_409_CONFLICT

def test_lock_fails_if_not_unlocked(integration_client):
    orchestrator = integration_client.app.state.orchestrator
    
    response = integration_client.post("/api/v1/vault/lock")

    assert response.status_code == status.HTTP_403_FORBIDDEN

    password_request = VaultPasswordRequest(
        password=SecretStr("Saturate-Heave8-Unfasten-Squealing")
    )
    
    response = integration_client.post(
        "/api/v1/vault/initialize", json=password_request.model_dump(mode="json")
    )

    assert response.status_code == status.HTTP_200_OK
    assert orchestrator.state == AppState.UNLOCKED
    
    lock_response = integration_client.post("/api/v1/vault/lock")

    assert lock_response.status_code == status.HTTP_200_OK
    assert orchestrator.state == AppState.LOCKED