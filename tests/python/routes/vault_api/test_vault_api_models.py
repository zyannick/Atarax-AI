import pytest
from pydantic import ValidationError
from ataraxai.routes.status import Status

from ataraxai.routes.vault_route.vault_api_models import (
    ConfirmationPhaseRequest,
    ConfirmationPhaseResponse,
    VaultPasswordRequest,
    VaultPasswordResponse,
    LockVaultResponse,
)

def test_confirmation_phase_request_valid():
    req = ConfirmationPhaseRequest(confirmation_phrase="confirm123")
    assert req.confirmation_phrase == "confirm123"

def test_confirmation_phase_request_missing_phrase():
    with pytest.raises(ValidationError):
        ConfirmationPhaseRequest()

def test_confirmation_phase_response_valid():
    resp = ConfirmationPhaseResponse(status=Status.SUCCESS, message="Confirmed")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Confirmed"

def test_confirmation_phase_response_missing_fields():
    with pytest.raises(ValidationError):
        ConfirmationPhaseResponse(status=Status.SUCCESS)

def test_vault_password_request_valid():
    req = VaultPasswordRequest(password="supersecret123")
    assert req.password.get_secret_value() == "supersecret123"

def test_vault_password_request_too_short():
    with pytest.raises(ValidationError):
        VaultPasswordRequest(password="short")

def test_vault_password_response_valid():
    resp = VaultPasswordResponse(status=Status.SUCCESS, message="Vault unlocked")
    assert resp.status == Status.SUCCESS
    assert resp.message == "Vault unlocked"

def test_lock_vault_response_valid():
    resp = LockVaultResponse(status=Status.FAILURE, message="Vault lock failed")
    assert resp.status == Status.FAILURE
    assert resp.message == "Vault lock failed"