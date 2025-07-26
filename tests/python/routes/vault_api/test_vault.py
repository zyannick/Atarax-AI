import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai.praxis.utils.app_state import AppState

# from ataraxai.routes.vault_route.vault import router_vault
from ataraxai.routes.status import Status

# from fastapi import FastAPI
from ataraxai.routes.rag_route.rag import get_unlocked_orchestrator
from ataraxai.routes.vault_route.vault import get_orchestrator

# from unittest import mock
from api import app




# @pytest.fixture
# def mock_unlocked_orchestrator():
#     """Fixture that provides a mock orchestrator for dependency injection"""
#     mock_orch = MagicMock()
#     app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch
#     yield mock_orch
#     app.dependency_overrides.clear()


# @pytest.fixture
# def mock_orchestrator_dep():
#     """Fixture that provides a mock orchestrator for the get_orchestrator dependency"""
#     mock_orch = MagicMock()
#     app.dependency_overrides[get_orchestrator] = lambda: mock_orch
#     yield mock_orch
#     app.dependency_overrides.clear()


class VaultPasswordRequest:
    def __init__(self, password):
        self.password = password


class ConfirmationPhaseRequest:
    def __init__(self, confirmation_phrase):
        self.confirmation_phrase = confirmation_phrase


class TestVaultInitialization:

    def test_initialize_vault_success(self, client):
        """Tests successful vault initialization from the FIRST_LAUNCH state."""
        # ARRANGE
        mock_orchestrator = client.app.state.orchestrator
        mock_orchestrator.state = AppState.FIRST_LAUNCH
        mock_orchestrator.initialize_new_vault.return_value = True

        payload = {"password": "a-strong-password"}

        # ACT
        response = client.post("/api/v1/vault/initialize", json=payload)

        # ASSERT
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == Status.SUCCESS
        assert data["message"] == "Vault initialized and unlocked."
        mock_orchestrator.initialize_new_vault.assert_called_once()

    # def test_initialize_vault_failure_short_password(self, client):
    #     mock_orchestrator = MagicMock()
    #     mock_orchestrator.state = AppState.FIRST_LAUNCH
    #     app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator

    #     payload = {"password": "test"}
    #     response = client.post("/api/v1/vault/initialize", json=payload)

    #     assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    #     app.dependency_overrides.clear()

#     def test_initialize_vault_failure(self, client):
#         moch_orchestrator = MagicMock()
#         moch_orchestrator.state = AppState.FIRST_LAUNCH
#         moch_orchestrator.initialize_new_vault.return_value = False
#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: moch_orchestrator

#         payload = {"password": "a-strong_password"}
#         response = client.post("/api/v1/vault/initialize", json=payload)

#         assert response.status_code == status.HTTP_200_OK
#         data = response.json()
#         assert data["status"] == Status.ERROR
#         assert data["message"] == "Failed to initialize vault."

#         app.dependency_overrides.clear()

#     def test_reinitialize_vault_success(self, client):

#         mock_orchestrator = MagicMock()
#         mock_orchestrator.reinitialize_vault.return_value = True
#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator

#         payload = {"confirmation_phrase": "confirm"}
#         response = client.post("/api/v1/vault/reinitialize", json=payload)

#         assert response.status_code == 200
#         data = response.json()
#         assert data["status"] == Status.SUCCESS
#         assert data["message"] == "Vault reinitialized and unlocked."
#         mock_orchestrator.reinitialize_vault.assert_called_once()
#         app.dependency_overrides.clear()

#     def test_reinitialize_vault_failure(self, client):
#         mock_orchestrator = MagicMock()
#         mock_orchestrator.reinitialize_vault.return_value = False
#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orchestrator

#         payload = {"confirmation_phrase": "confirm"}
#         response = client.post("/api/v1/vault/reinitialize", json=payload)

#         assert response.status_code == status.HTTP_200_OK
#         data = response.json()
#         assert data["status"] == Status.ERROR
#         assert data["message"] == "Failed to reinitialize vault."
#         mock_orchestrator.reinitialize_vault.assert_called_once()
#         app.dependency_overrides.clear()


# class TestVaultLocking:

#     def test_unlock_success(self, client):
#         mock_orch = MagicMock()
#         mock_orch.state = AppState.LOCKED
#         mock_orch.unlock.return_value = True

#         app.dependency_overrides[get_orchestrator] = lambda: mock_orch

#         payload = {"password": "testpass"}
#         response = client.post("/api/v1/vault/unlock", json=payload)

#         assert (
#             response.status_code == status.HTTP_200_OK
#         ), f"Expected 200 OK, got {response.status_code} with message: {response.text}"
#         assert response.json()["status"] == Status.SUCCESS
#         mock_orch.unlock.assert_called_once()

#         app.dependency_overrides.clear()

#     def test_unlock_conflict(self, client):
#         mock_orch = MagicMock()
#         mock_orch.state = AppState.UNLOCKED

#         app.dependency_overrides[get_orchestrator] = lambda: mock_orch

#         payload = {"password": "testpass"}
#         response = client.post("/api/v1/vault/unlock", json=payload)

#         assert response.status_code == status.HTTP_409_CONFLICT
#         mock_orch.unlock.assert_not_called()

#         app.dependency_overrides.clear()

#     def test_unlock_failure(self, client):
#         mock_orch = MagicMock()
#         mock_orch.state = AppState.LOCKED
#         mock_orch.unlock.return_value = False

#         app.dependency_overrides[get_orchestrator] = lambda: mock_orch

#         payload = {"password": "testpass"}
#         response = client.post("/api/v1/vault/unlock", json=payload)

#         assert response.status_code == status.HTTP_200_OK
#         data = response.json()
#         assert data["status"] == Status.ERROR
#         assert data["message"] == "Failed to unlock vault. Incorrect password."
#         mock_orch.unlock.assert_called_once()

#         app.dependency_overrides.clear()

#     def test_lock_success(self, client):
#         mock_orch = MagicMock()
#         mock_orch.lock.return_value = True

#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch

#         response = client.post("/api/v1/vault/lock")

#         assert response.status_code == status.HTTP_200_OK
#         assert response.json()["status"] == Status.SUCCESS
#         mock_orch.lock.assert_called_once()

#         app.dependency_overrides.clear()

#     def test_lock_failure(self, client):
#         mock_orch = MagicMock()
#         mock_orch.lock.return_value = False

#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch

#         response = client.post("/api/v1/vault/lock")

#         assert response.status_code == status.HTTP_200_OK
#         data = response.json()
#         assert data["status"] == Status.ERROR
#         assert data["message"] == "Failed to lock vault."
#         mock_orch.lock.assert_called_once()

#         app.dependency_overrides.clear()


# class TestVaultEdgeCases:

#     def test_initialize_vault_invalid_payload(self, client):
#         payload = {}
#         response = client.post("/api/v1/vault/initialize", json=payload)
#         assert (
#             response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#         ), f"Expected 422, got {response.status_code} with message: {response.text}"

#     def test_reinitialize_vault_invalid_payload(self, client):
#         payload = {}
#         response = client.post("/api/v1/vault/reinitialize", json=payload)
#         assert (
#             response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#         ), f"Expected 422, got {response.status_code} with message: {response.text}"

#     def test_unlock_invalid_payload(self, client):
#         payload = {}
#         response = client.post("/api/v1/vault/unlock", json=payload)

#         assert (
#             response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#         ), f"Expected 422, got {response.status_code} with message: {response.text}"

#     def test_initialize_vault_exception(self, client):
#         mock_orch = MagicMock()
#         mock_orch.initialize_new_vault.side_effect = Exception("Database error")

#         app.dependency_overrides[get_unlocked_orchestrator] = lambda: mock_orch

#         payload = {"password": "testpass"}
#         response = client.post("/api/v1/vault/initialize", json=payload)

#         assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

#         app.dependency_overrides.clear()
