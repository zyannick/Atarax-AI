from typing import Any
from fastapi import HTTPException, Request, WebSocketDisconnect
from fastapi import WebSocket
from fastapi.params import Depends
from ataraxai.gateway.request_manager import RequestManager
from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from fastapi import status
from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.constant_messages import Messages
from ataraxai.praxis.katalepsis import Katalepsis

def get_orchestrator(request: Request) -> AtaraxAIOrchestrator:
    """
    Retrieve the AtaraxAIOrchestrator instance from the application's state.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        AtaraxAIOrchestrator: The orchestrator instance stored in the application's state.
    """
    return request.app.state.orchestrator

def get_orchestrator_ws(websocket: WebSocket) -> AtaraxAIOrchestrator:
    """
    Retrieve the AtaraxAIOrchestrator instance from the application's state via the provided WebSocket.

    Args:
        websocket (WebSocket): The WebSocket connection from which to access the application's state.

    Returns:
        AtaraxAIOrchestrator: The orchestrator instance stored in the application's state.
    """
    return websocket.app.state.orchestrator

def get_request_manager(request: Request) -> RequestManager:
    """
    Dependency to get the RequestManager instance from the FastAPI application state.
    
    Args:
        request (Request): The FastAPI request object.

    Returns:
        RequestManager: The RequestManager instance from the application state.
    """
    return request.app.state.request_manager

def get_gatewaye_task_manager(request: Request) -> GatewayTaskManager:
    """
    Dependency to get the TaskManager instance from the FastAPI application state.
    
    Args:
        request (Request): The FastAPI request object.

    Returns:
        GatewayTaskManager: The GatewayTaskManager instance from the application state.

    Returns:
        GatewayTaskManager: The GatewayTaskManager instance from the application state.
    """
    return request.app.state.gateway_task_manager


def get_unlocked_orchestrator(
    orch: Any = Depends(get_orchestrator),
) -> AtaraxAIOrchestrator:
    """
    Dependency function that retrieves the current AtaraxAIOrchestrator instance only if its state is UNLOCKED.

    Raises:
        HTTPException: If the orchestrator's state is not UNLOCKED, returns a 403 Forbidden error with a vault locked message.

    Returns:
        AtaraxAIOrchestrator: The unlocked orchestrator instance.
    """
    if orch.state != AppState.UNLOCKED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=Messages.VAULT_LOCKED
        )
    return orch

def get_unlocked_orchestrator_ws(
    orch: Any = Depends(get_orchestrator_ws),
) -> AtaraxAIOrchestrator:
    """
    Dependency that ensures the orchestrator WebSocket is in the UNLOCKED state.

    Retrieves the orchestrator WebSocket instance and checks its state. If the orchestrator
    is not unlocked, raises a WebSocketDisconnect with an appropriate policy violation code
    and reason. Otherwise, returns the orchestrator instance.

    Args:
        orch (AtaraxAIOrchestrator): The orchestrator WebSocket instance, injected via dependency.

    Returns:
        AtaraxAIOrchestrator: The unlocked orchestrator WebSocket instance.

    Raises:
        WebSocketDisconnect: If the orchestrator is not in the UNLOCKED state.
    """
    if orch.state != AppState.UNLOCKED:
        raise WebSocketDisconnect(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=Messages.VAULT_LOCKED,
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