import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from ataraxai.routes.vault import router_vault
from ataraxai.routes.status import Status

class VaultPasswordRequest:
    def __init__(self, password):
        self.password = password

class ConfirmationPhaseRequest:
    def __init__(self, confirmation_phrase):
        self.confirmation_phrase = confirmation_phrase

client = TestClient(router_vault)

@pytest.fixture
def orch_mock():
    return MagicMock()

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_initialize_vault_success(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.initialize_new_vault.return_value = True
    get_unlocked_orchestrator.return_value = orch

    payload = {"password": "testpass"}
    response = client.post("/api/v1/vault/initialize", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_initialize_vault_failure(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.initialize_new_vault.return_value = False
    get_unlocked_orchestrator.return_value = orch

    payload = {"password": "testpass"}
    response = client.post("/api/v1/vault/initialize", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_reinitialize_vault_success(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.reinitialize_vault.return_value = True
    get_unlocked_orchestrator.return_value = orch

    payload = {"confirmation_phrase": "confirm"}
    response = client.post("/api/v1/vault/reinitialize", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_reinitialize_vault_failure(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.reinitialize_vault.return_value = False
    get_unlocked_orchestrator.return_value = orch

    payload = {"confirmation_phrase": "confirm"}
    response = client.post("/api/v1/vault/reinitialize", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

@patch("ataraxai.routes.vault.get_orchestrator")
@patch("ataraxai.routes.vault.SecureString")
def test_unlock_success(SecureString, get_orchestrator):
    orch = MagicMock()
    orch.state = "LOCKED"
    orch.unlock.return_value = True
    get_orchestrator.return_value = orch
    SecureString.return_value = b"testpass"

    payload = {"password": "testpass"}
    response = client.post("/api/v1/vault/unlock", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

@patch("ataraxai.routes.vault.get_orchestrator")
def test_unlock_conflict(get_orchestrator):
    orch = MagicMock()
    orch.state = "UNLOCKED"
    get_orchestrator.return_value = orch

    payload = {"password": "testpass"}
    response = client.post("/api/v1/vault/unlock", json=payload)
    assert response.status_code == status.HTTP_409_CONFLICT

@patch("ataraxai.routes.vault.get_orchestrator")
@patch("ataraxai.routes.vault.SecureString")
def test_unlock_failure(SecureString, get_orchestrator):
    orch = MagicMock()
    orch.state = "LOCKED"
    orch.unlock.return_value = False
    get_orchestrator.return_value = orch
    SecureString.return_value = b"testpass"

    payload = {"password": "testpass"}
    response = client.post("/api/v1/vault/unlock", json=payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_lock_success(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.lock.return_value = True
    get_unlocked_orchestrator.return_value = orch

    response = client.post("/api/v1/vault/lock")
    assert response.status_code == 200
    assert response.json()["status"] == Status.SUCCESS

@patch("ataraxai.routes.vault.get_unlocked_orchestrator")
def test_lock_failure(get_unlocked_orchestrator):
    orch = MagicMock()
    orch.lock.return_value = False
    get_unlocked_orchestrator.return_value = orch

    response = client.post("/api/v1/vault/lock")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR