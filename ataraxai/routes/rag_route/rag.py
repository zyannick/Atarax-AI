from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from fastapi import Depends

from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager, RequestPriority
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.rag_route.rag_api_models import (
    CheckManifestResponse,
    DirectoriesAdditionResponse,
    DirectoriesRemovalResponse,
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
    RebuildIndexResponse,
)
from ataraxai.routes.status import StatusResponse
from ataraxai.routes.status import TaskStatus as Status

logger = AtaraxAILogger("ataraxai.praxis.rag").get_logger()


router_rag = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router_rag.post("/rebuild_index", response_model=RebuildIndexResponse)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Rebuild Index", logger=logger)
async def rebuild_index(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> RebuildIndexResponse:
    """
    Endpoint to rebuild the RAG (Retrieval-Augmented Generation) index.

    This endpoint triggers the orchestrator to rebuild the RAG index, which may be necessary after data updates or changes.
    On success, it returns a response indicating the index was rebuilt successfully.
    If an error occurs during the rebuild process, an HTTP 500 error is raised with a relevant message.

    Args:
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        RebuildIndexResponse: Response object containing the status and message of the rebuild operation.

    Raises:
        HTTPException: If an error occurs during the index rebuild process.
    """
    rag_manager = await orch.get_rag_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Rebuild Index",
        func=rag_manager.rebuild_index,
        priority=RequestPriority.HIGH,
    )
    task_id = task_manager.create_task(future)  # type: ignore
    return RebuildIndexResponse(
        status=Status.SUCCESS,
        message="Index rebuild started successfully.",
        result=task_id,
    )


@router_rag.get("/rebuild_index/{task_id}", response_model=RebuildIndexResponse)
async def get_rebuild_index_result(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> RebuildIndexResponse:
    """
    Endpoint to retrieve the result of a previously initiated index rebuild operation.

    Args:
        task_id (str): The ID of the task for which the rebuild index result is requested.
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        RebuildIndexResponse: Response object containing the status and message of the rebuild operation.

    Raises:
        HTTPException: If an error occurs while retrieving the task status.
    """
    result = task_manager.get_task_status(task_id)
    if result is None:
        return RebuildIndexResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found.",
            result=None,
        )
    return RebuildIndexResponse(
        status=Status.SUCCESS,
        message="Rebuild index operation completed successfully.",
        result=result,
    )


@router_rag.delete("/rebuild_index/{task_id}", response_model=RebuildIndexResponse)
async def cancel_rebuild_index(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> RebuildIndexResponse:
    """
    Endpoint to cancel a previously initiated index rebuild operation.

    Args:
        task_id (str): The ID of the task to be canceled.
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        RebuildIndexResponse: Response object indicating the status and message of the cancellation operation.
    """
    was_cancelled = task_manager.cancel_task(task_id)
    if not was_cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found or could not be canceled.",
        )
    return RebuildIndexResponse(
        status=Status.SUCCESS,
        message=f"Task with ID {task_id} has been canceled successfully.",
        result=None,
    )


@router_rag.get("/check_manifest", response_model=CheckManifestResponse)
async def check_manifest(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
):
    rag_manager = await orch.get_rag_manager()
    is_valid = await rag_manager.check_manifest_validity()
    if is_valid:
        return CheckManifestResponse(
            status=Status.SUCCESS, message="Manifest is valid."
        )
    else:
        return CheckManifestResponse(
            status=Status.ERROR, message="Manifest is invalid or missing."
        )


@router_rag.get("/health_check", response_model=StatusResponse)
async def health_check(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
):
    rag_manager = await orch.get_rag_manager()
    is_healthy = await rag_manager.health_check()
    return StatusResponse(
        status=Status.SUCCESS if is_healthy else Status.ERROR,
        message=(
            "RAG manager is healthy." if is_healthy else "RAG manager is not healthy."
        ),
    )


@router_rag.post("/add_directories", response_model=DirectoriesAdditionResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Add Directories", logger=logger)
async def add_directory(
    request: DirectoriesToAddRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> DirectoriesAdditionResponse:
    """
    Endpoint to add directories for RAG (Retrieval-Augmented Generation) indexing.

    This endpoint accepts a list of directory paths to be watched and indexed in the background.
    It schedules the addition of these directories as a background task and returns a success message if the task is scheduled successfully.
    If an error occurs while scheduling the task, an HTTP 500 error is raised.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency for managing background tasks.
        request (DirectoriesToAddRequest): Request body containing the list of directories to add.
        orch: Dependency-injected orchestrator instance.

    Returns:
        DirectoriesAdditionResponse: Response indicating the status and message of the operation.

    Raises:
        HTTPException: If an error occurs while adding directories, returns HTTP 500 with an error message.
    """
    rag_manager = await orch.get_rag_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Add Directories",
        func=rag_manager.add_watch_directories,
        directories=request.directories,
        priority=RequestPriority.HIGH,
    )
    task_id = task_manager.create_task(future)  # type: ignore

    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule directory addition task.",
        )
    return DirectoriesAdditionResponse(
        status=Status.SUCCESS,
        message="Directories started to be added for indexing.",
        result=task_id,
    )


@router_rag.post("/remove_directories", response_model=DirectoriesRemovalResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Remove Directories", logger=logger)
async def remove_directory(
    request: DirectoriesToRemoveRequest,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
) -> DirectoriesRemovalResponse:
    """
    Endpoint to remove directories from RAG (Retrieval-Augmented Generation) indexing.

    This endpoint accepts a list of directory paths to be removed from the indexing process.
    It schedules the removal of these directories as a background task and returns a success message if the task is scheduled successfully.
    If an error occurs while scheduling the task, an HTTP 500 error is raised.

    Args:
        request (DirectoriesToRemoveRequest): Request body containing the list of directories to remove.
        orch: Dependency-injected orchestrator instance.

    Returns:
        DirectoriesRemovalResponse: Response indicating the status and message of the operation.

    Raises:
        HTTPException: If an error occurs while removing directories, returns HTTP 500 with an error message.
    """
    rag_manager = await orch.get_rag_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Remove Directories",
        func=rag_manager.remove_watch_directories,
        directories=request.directories,
        priority=RequestPriority.HIGH,
    )
    task_id = task_manager.create_task(future)  # type: ignore

    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule directory removal task.",
        )
    return DirectoriesRemovalResponse(
        status=Status.SUCCESS,
        message="Directories started to be removed from indexing.",
        result=task_id,
    )
