from typing import Any
from fastapi import HTTPException, Request
from fastapi.params import Depends
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from fastapi import status
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.constant_messages import Messages
from ataraxai.praxis.katalepsis import Katalepsis

def get_orchestrator(request: Request) -> AtaraxAIOrchestrator:
    return request.app.state.orchestrator


def get_unlocked_orchestrator(
    orch: Any = Depends(get_orchestrator),
) -> AtaraxAIOrchestrator:
    print(f"Orchestrator state: {orch.state}")  # Debugging line, remove in production
    if orch.state != AppState.UNLOCKED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=Messages.VAULT_LOCKED
        )
    return orch

def get_katalepsis_monitor(request: Request) -> Katalepsis:
    """
    Dependency to get the Katalepsis instance from the FastAPI application state.
    
    Args:
        request (Request): The FastAPI request object.

    Returns:
        Katalepsis: The Katalepsis instance from the application state.
    """
    return request.app.state.katalepsis