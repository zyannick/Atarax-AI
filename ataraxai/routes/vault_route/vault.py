from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends

from ataraxai.hegemonikon_py import SecureString  # type: ignore
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.praxis.utils.vault_manager import (
    UnlockResult,
    VaultInitializationStatus,
    VaultUnlockStatus,
)
from ataraxai.routes.dependency_api import get_orchestrator, get_unlocked_orchestrator
from ataraxai.routes.status import Status
from ataraxai.routes.vault_route.vault_api_models import (
    ConfirmationPhaseRequest,
    ConfirmationPhaseResponse,
    LockVaultResponse,
    VaultPasswordRequest,
    VaultPasswordResponse,
)

logger = AtaraxAILogger("ataraxai.praxis.vault").get_logger()


router_vault = APIRouter(prefix="/api/v1/vault", tags=["Vault"])


@router_vault.post("/initialize", response_model=VaultPasswordResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Initialize Vault", logger=logger)
async def initialize_vault(
    request: VaultPasswordRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> VaultPasswordResponse:
    """
    Initializes a new vault with the provided password.

    This endpoint creates and unlocks a new vault using the password supplied in the request.
    If the vault is successfully initialized, it returns a success status and message.
    If initialization fails, it logs the error and raises an HTTP 500 error.

    Args:
        request (VaultPasswordRequest): The request body containing the password for vault initialization.
        orch: The unlocked orchestrator dependency.

    Returns:
        VaultPasswordResponse: The response containing the status and message of the operation.

    Raises:
        HTTPException: If a critical error occurs during vault initialization.
    """
    logger.info("Initializing vault with provided password.")
    state = await orch.get_state()
    if state != AppState.FIRST_LAUNCH:
        logger.error("Vault is already initialized.")
        return VaultPasswordResponse(
            status=Status.ERROR, message="Vault is already initialized."
        )
    password_bytes = request.password.get_secret_value().encode("utf-8")
    success: VaultInitializationStatus = orch.initialize_new_vault(SecureString(password_bytes))  # type: ignore

    if success == VaultInitializationStatus.SUCCESS:
        return VaultPasswordResponse(
            status=Status.SUCCESS, message="Vault initialized and unlocked."
        )
    return VaultPasswordResponse(
        status=Status.FAILED, message="Failed to initialize vault."
    )


@router_vault.post("/reinitialize", response_model=ConfirmationPhaseResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Reinitialize Vault", logger=logger)
async def reinitialize_vault(
    request: ConfirmationPhaseRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> ConfirmationPhaseResponse:
    """
    Reinitializes the vault using the provided confirmation phrase.

    This endpoint attempts to reinitialize and unlock the vault. If the operation is successful,
    it returns a confirmation response indicating success. If the operation fails, it logs the error
    and raises an HTTP 500 error.

    Args:
        request (ConfirmationPhaseRequest): The request body containing the confirmation phrase.
        orch: The unlocked orchestrator dependency.

    Returns:
        ConfirmationPhaseResponse: The response indicating the status of the reinitialization.

    Raises:
        HTTPException: If a critical error occurs during vault reinitialization.
    """
    success = orch.reinitialize_vault(request.confirmation_phrase)

    if success:
        return ConfirmationPhaseResponse(
            status=Status.SUCCESS,
            message="Vault reinitialized for first launch. We need to set a new password.",
        )
    else:
        return ConfirmationPhaseResponse(
            status=Status.ERROR, message="Failed to reinitialize vault."
        )


@router_vault.post("/unlock", response_model=VaultPasswordResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Unlock Vault", logger=logger)
async def unlock(
    request: VaultPasswordRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_orchestrator)],
) -> VaultPasswordResponse:
    """
    Unlocks the vault using the provided password.

    This endpoint receives a password in the request body and attempts to unlock the vault.
    If the password is correct and the vault is successfully unlocked, it returns a success status.
    If unlocking fails, it logs an error and raises an HTTP 500 error.

    Args:
        request (VaultPasswordRequest): The request body containing the vault password.
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        VaultPasswordResponse: The response indicating the status of the unlock operation.

    Raises:
        HTTPException: If unlocking the vault fails, raises a 500 Internal Server Error.
    """
    state = await orch.get_state()
    if state != AppState.LOCKED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Vault is not locked."
        )

    password_bytes = request.password.get_secret_value().encode("utf-8")
    unlock_result: UnlockResult = await orch.unlock(SecureString(password_bytes))

    if unlock_result.status == VaultUnlockStatus.SUCCESS:
        return VaultPasswordResponse(status=Status.SUCCESS, message="Vault unlocked.")
    elif unlock_result.status == VaultUnlockStatus.INCORRECT_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password provided for vault unlock.",
        )
    elif unlock_result.status == VaultUnlockStatus.ALREADY_UNLOCKED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Vault is already unlocked.",
        )
    elif unlock_result.status == VaultUnlockStatus.UNINITIALIZED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vault is uninitialized. Please initialize it first.",
        )
    else:
        logger.error("Failed to unlock vault due to an unexpected error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unlock vault due to an unexpected error.",
        )


@router_vault.post("/lock", response_model=LockVaultResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Lock Vault", logger=logger)
async def lock(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
) -> LockVaultResponse:
    """
    Locks the vault by invoking the orchestrator's lock method.

    This endpoint attempts to lock the vault. If the operation is successful, it returns a response indicating success.
    If the operation fails, it logs an error and raises an HTTP 500 error.

    Args:
        orch: The unlocked orchestrator dependency, injected by FastAPI.

    Returns:
        LockVaultResponse: Response object indicating the status and message of the lock operation.

    Raises:
        HTTPException: If the vault fails to lock, raises an HTTP 500 error with a relevant message.
    """
    success = await orch.lock()

    if success:
        return LockVaultResponse(
            status=Status.SUCCESS, message="Vault locked successfully."
        )
    else:
        logger.error("Failed to lock vault.")
        return LockVaultResponse(status=Status.ERROR, message="Failed to lock vault.")
