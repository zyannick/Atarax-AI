from fastapi import APIRouter, BackgroundTasks
from fastapi.params import Depends
from ataraxai.routes.rag_route.rag_api_models import (
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
    RebuildIndexResponse,
    ScanAndIndexResponse,
    DirectoriesAdditionResponse,
    DirectoriesRemovalResponse,
    CheckManifestResponse,
)
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor


logger = AtaraxAILogger("ataraxai.praxis.rag").get_logger()


router_rag = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router_rag.get("/check_manifest", response_model=CheckManifestResponse)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("Check Manifest", logger=logger)
async def check_manifest(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> CheckManifestResponse:  # type: ignore
    """
    Endpoint to check the validity of the RAG manifest.

    This endpoint verifies whether the manifest used by the RAG orchestrator is valid or missing.
    Returns a response indicating the status and a message describing the result.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        CheckManifestResponse: An object containing the status (SUCCESS or ERROR) and a descriptive message.

    Raises:
        HTTPException: If an error occurs during the manifest check, returns a 500 Internal Server Error with details.
    """
    is_valid = orch.rag.check_manifest_validity()
    if is_valid:
        return CheckManifestResponse(
            status=Status.SUCCESS, message="Manifest is valid."
        )
    else:
        return CheckManifestResponse(
            status=Status.ERROR,
            message="Manifest is invalid or missing.",
        )


@router_rag.post("/rebuild_index", response_model=RebuildIndexResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Rebuild Index", logger=logger)
async def rebuild_index(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> RebuildIndexResponse:  # type: ignore
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

    orch.rag.rebuild_index()
    return RebuildIndexResponse(
        status=Status.SUCCESS, message="Index rebuilt successfully."
    )


@router_rag.post("/scan_and_index", response_model=ScanAndIndexResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Scan and Index", logger=logger)
async def scan_and_index(background_tasks: BackgroundTasks, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> ScanAndIndexResponse:  # type: ignore
    """
    Starts a background task to perform an initial scan and indexing operation.

    This endpoint triggers the orchestrator's RAG (Retrieval-Augmented Generation) initial scan and indexing process asynchronously.
    The operation is executed in the background, allowing the API to return immediately with a success message.

    Args:
        background_tasks (BackgroundTasks): FastAPI dependency for managing background tasks.
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        ScanAndIndexResponse: Response indicating that the scan and indexing process has started.

    Raises:
        HTTPException: If an error occurs while starting the scan and indexing process, returns a 500 Internal Server Error.
    """
    background_tasks.add_task(orch.rag.perform_initial_scan, [])
    return ScanAndIndexResponse(
        status=Status.SUCCESS,
        message="Scan and indexing started in the background.",
    )


@router_rag.post("/add_directories", response_model=DirectoriesAdditionResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Add Directories", logger=logger)
async def add_directory(
    background_tasks: BackgroundTasks, request: DirectoriesToAddRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)  # type: ignore
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
    background_tasks.add_task(orch.rag.add_watch_directories, request.directories)
    return DirectoriesAdditionResponse(
        status=Status.SUCCESS,
        message=f"Directories '{', '.join(request.directories)}' added for indexing.",
    )


@router_rag.post("/remove_directories", response_model=DirectoriesRemovalResponse)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Remove Directories", logger=logger)
async def remove_directory(
    background_tasks: BackgroundTasks, request: DirectoriesToRemoveRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)  # type: ignore
) -> DirectoriesRemovalResponse:
    """
    Removes specified directories from the RAG indexing process asynchronously.

    This endpoint schedules the removal of each directory listed in the request from the
    orchestrator's watch list using background tasks. If successful, returns a response
    indicating the directories have been removed from indexing. In case of failure,
    logs the error and raises an HTTP 500 exception.

    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager for scheduling directory removal.
        request (DirectoriesToRemoveRequest): Request body containing a list of directories to remove.
        orch: Dependency-injected orchestrator instance (unlocked).

    Returns:
        DirectoriesRemovalResponse: Response indicating the status and message of the removal operation.

    Raises:
        HTTPException: If an error occurs during the removal process, returns HTTP 500 with error details.
    """
    background_tasks.add_task(orch.rag.remove_watch_directories, request.directories)
    return DirectoriesRemovalResponse(
        status=Status.SUCCESS,
        message=f"Directories '{', '.join(request.directories)}' removed from indexing.",
    )
