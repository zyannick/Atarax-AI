from fastapi import APIRouter, FastAPI, HTTPException, BackgroundTasks, Request
from contextlib import asynccontextmanager
from fastapi.params import Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from fastapi import FastAPI
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory

from ataraxai.praxis.utils.app_state import AppState
from ataraxai.routes.chat import (
    CreateProjectRequest,
    CreateProjectResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    DeleteProjectRequest,
    DeleteProjectResponse,
    DeleteSessionResponse,
)
from ataraxai.routes.vault import (
    ConfirmationPhaseRequest,
    ConfirmationPhaseResponse,
    VaultPasswordRequest,
    VaultPasswordResponse,
    LockVaultResponse,
)
from ataraxai.routes.rag import (
    DirectoriesToAddRequest,
    DirectoriesToRemoveRequest,
    RebuildIndexResponse,
    ScanAndIndexResponse,
    DirectoriesAdditionResponse,
    DirectoriesRemovalResponse,
    CheckManifestResponse,
)
from ataraxai.routes.status import StatusResponse, Status
from ataraxai.praxis.utils.exceptions import AtaraxAIError, AtaraxAILockError
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger


from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from ataraxai.praxis.katalepsis import Katalepsis


class HTTPStatus:
    OK = 200
    NOT_FOUND = 404
    FORBIDDEN = 403
    INTERNAL_SERVER_ERROR = 500
    CONFLICT = 409


class Messages:
    VAULT_LOCKED = "Vault is locked"
    VAULT_UNLOCKED = "Vault is already unlocked"
    PROJECT_NOT_FOUND = "Project not found"
    SESSION_NOT_FOUND = "Session not found"
    OPERATION_FAILED = "An error occurred while processing the request"


logger = AtaraxAILogger("ataraxai.praxis.api")

katalepsis_monitor = Katalepsis()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orchestrator = AtaraxAIOrchestratorFactory.create_orchestrator()
    yield
    logger.info("API is shutting down. Closing orchestrator resources.")
    app.state.orchestrator.shutdown()

def get_orchestrator(request: Request) -> AtaraxAIOrchestrator:
    return request.app.state.orchestrator


app = FastAPI(
    title="AtaraxAI API",
    description="API for the AtaraxAI Local Assistant Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_unlocked_orchestrator(orch: Any = Depends(get_orchestrator)) -> AtaraxAIOrchestrator:
    if orch.state != AppState.UNLOCKED:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Vault is locked.")
    return orch


router_vault = APIRouter(prefix="/api/v1/vault", tags=["Vault"])
router_chat = APIRouter(prefix="/api/v1", tags=["Chat"])
router_rag = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router_chat.post("/projects", response_model=CreateProjectResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def create_new_project(
    project_data: CreateProjectRequest,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
):
    """
    Create a new project.

    This endpoint creates a new project using the provided project data.
    It returns the details of the created project upon success.

    Args:
        project_data (CreateProjectRequest): The data required to create a new project.
        orch: The orchestrator dependency, injected by FastAPI.

    Returns:
        CreateProjectResponse: The response containing the created project's details.

    Raises:
        HTTPException: If an error occurs during project creation, returns a 500 Internal Server Error.
    """
    try:
        project = orch.chat.create_project(
            name=project_data.name, description=project_data.description
        )

        return CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
    except Exception as e:
        logger.error(f"Failed to create project {project_data.name}: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating the project: {project_data.name}",
        )


@router_chat.delete("/projects/{project_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def delete_project(project_id: uuid.UUID, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Deletes a project by its unique project ID.

    Args:
        project_id (uuid.UUID): The unique identifier of the project to delete.
        orch: The orchestrator dependency, injected by FastAPI.

    Returns:
        StatusResponse: A response object indicating the status and message of the deletion operation.

    Raises:
        HTTPException:
            - 404 NOT FOUND if the project does not exist.
            - 500 INTERNAL SERVER ERROR if an error occurs during deletion.
    """
    project = orch.chat.get_project(project_id)
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Project not found"
        )

    try:
        orch.chat.delete_project(project_id)
        return StatusResponse(
            status=Status.SUCCESS, message=f"Project {project_id} deleted successfully."
        )
    except Exception as e:
        logger.error(f"Failed to delete project {project_id}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the project: {project_id}",
        )


@router_chat.get("/projects/{project_id}", response_model=CreateProjectResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def get_project(project_id: uuid.UUID, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Retrieve a project by its unique identifier.

    Args:
        project_id (uuid.UUID): The unique identifier of the project to retrieve.
        orch: Dependency injection for the unlocked orchestrator.

    Returns:
        CreateProjectResponse: The response model containing project details (ID, name, description).

    Raises:
        HTTPException: If the project with the specified ID is not found, returns a 404 NOT FOUND error.
    """
    project = orch.chat.get_project(project_id)
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Project not found"
        )
    return CreateProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@router_chat.get("/projects/list", response_model=List[CreateProjectResponse])
@katalepsis_monitor.instrument_api  # type: ignore
async def list_projects(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Handles GET requests to "/v1/projects/list" and returns a list of projects.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        List[CreateProjectResponse]: A list of project details, each containing project_id, name, and description.

    Raises:
        Any exceptions raised by the orchestrator's list_projects method.

    Decorators:
        @app.get: Registers the endpoint with FastAPI.
        @katalepsis_monitor.instrument_api: Instruments the API for monitoring.
    """
    projects = orch.chat.list_projects()
    return [
        CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
        for project in projects
    ]


@router_chat.get(
    "/projects/{project_id}/sessions", response_model=List[CreateSessionResponse]
)
@katalepsis_monitor.instrument_api  # type: ignore
async def list_sessions(project_id: uuid.UUID, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Retrieve a list of chat sessions for a given project.

    Args:
        project_id (uuid.UUID): The unique identifier of the project.
        orch: The unlocked orchestrator dependency (injected by FastAPI).

    Returns:
        List[CreateSessionResponse]: A list of session response objects, each containing
            the session ID, title, and associated project ID.

    Raises:
        HTTPException: If the project does not exist or access is denied.

    Endpoint:
        GET /v1/projects/{project_id}/sessions

    Instrumentation:
        katalepsis_monitor.instrument_api is used to monitor this API endpoint.
    """
    sessions = orch.chat.list_sessions(project_id)
    return [
        CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project_id
        )
        for session in sessions
    ]


@router_chat.post("/new_session", response_model=CreateSessionResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def create_new_session(session_data: CreateSessionRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Creates a new chat session for a given project.

    Args:
        session_data (CreateSessionRequest): The request body containing the project ID and optional session title.
        orch: The unlocked orchestrator dependency, injected by FastAPI.

    Returns:
        CreateSessionResponse: The response containing the new session's ID, title, and project ID.

    Raises:
        HTTPException: If an error occurs during session creation, returns a 500 Internal Server Error with details.
    """
    try:
        session = orch.chat.create_session(
            project_id=session_data.project_id, title=session_data.title
        )

        return CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project_id
        )
    except Exception as e:
        logger.error(f"Failed to create session for project {session_data.project_id}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Unable to create session. An error occurred: {str(e)}",
        )


@router_chat.delete("/sessions/{session_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def delete_session(session_id: uuid.UUID, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Deletes a chat session by its session ID.

    Args:
        session_id (uuid.UUID): The unique identifier of the session to delete.
        orch: The unlocked orchestrator dependency, used to access chat sessions.

    Raises:
        HTTPException: If the session with the given ID is not found (404 Not Found).

    Returns:
        StatusResponse: An object indicating the success status and a message confirming deletion.
    """
    session = orch.chat.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Session not found"
        )

    orch.chat.delete_session(session_id)
    return StatusResponse(
        status=Status.SUCCESS, message=f"Session {session_id} deleted successfully."
    )


@router_chat.get("/sessions/{session_id}", response_model=CreateSessionResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def get_session(session_id: uuid.UUID, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Retrieve a chat session by its unique session ID.

    Args:
        session_id (uuid.UUID): The unique identifier of the session to retrieve.
        orch: The orchestrator dependency, injected by FastAPI.

    Returns:
        CreateSessionResponse: An object containing the session's ID, title, and project ID.

    Raises:
        HTTPException: If the session with the given ID is not found, raises a 404 Not Found error.
    """
    session = orch.chat.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Session not found"
        )
    return CreateSessionResponse(
        session_id=session.id, title=session.title, project_id=session.project_id
    )


@router_chat.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def send_message(
    session_id: uuid.UUID, message_data: ChatMessageRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)  # type: ignore
):
    """
    Handles sending a user message to a chat session and returns the assistant's response.

    Args:
        session_id (uuid.UUID): The unique identifier of the chat session.
        message_data (ChatMessageRequest): The user's message payload.
        orch: The orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        ChatMessageResponse: The assistant's response and session ID.

    Raises:
        HTTPException:
            - 404 NOT FOUND if the session does not exist.
            - 500 INTERNAL SERVER ERROR if an error occurs while sending the message.
    """
    session = orch.chat.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Session not found"
        )

    try:
        response = orch.chat.add_message(
            session_id=session_id, role="user", content=message_data.user_query
        )
        return ChatMessageResponse(
            assistant_response=response.content,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Failed to send message in session {session_id}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while sending the message: {str(e)}",
        )


@app.get("/v1/status", response_model=StatusResponse)
async def get_state(orch : AtaraxAIOrchestrator = Depends(get_orchestrator)) -> StatusResponse: # type: ignore
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"AtaraxAI is currently in state: {orch.state.name}",
    )


@router_vault.post("/initialize", response_model=VaultPasswordResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def initialize_vault(request: VaultPasswordRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> VaultPasswordResponse:  # type: ignore
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
    success = orch.initialize_new_vault(request.password)

    if success:
        return VaultPasswordResponse(
            status=Status.SUCCESS, message="Vault initialized and unlocked."
        )
    else:
        logger.error("Failed to initialize vault.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="A critical error occurred during vault initialization.",
        )


@router_vault.post("/reinitialize", response_model=ConfirmationPhaseResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def reinitialize_vault(
    request: ConfirmationPhaseRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)  # type: ignore
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
            status=Status.SUCCESS, message="Vault reinitialized and unlocked."
        )
    else:
        logger.error(f"Failed to reinitialize vault: {request.confirmation_phrase}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="A critical error occurred during vault reinitialization.",
        )


@router_vault.post("/unlock", response_model=VaultPasswordResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def unlock(request: VaultPasswordRequest, orch: AtaraxAIOrchestrator = Depends(get_orchestrator)) -> VaultPasswordResponse:  # type: ignore
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
    if orch.state != AppState.LOCKED:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail="Vault is not locked.")

    success = orch.unlock(request.password)

    if success:
        return VaultPasswordResponse(status=Status.SUCCESS, message="Vault unlocked.")
    else:
        logger.error("Failed to unlock vault.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="A critical error occurred during vault unlocking.",
        )


@router_vault.post("/lock", response_model=LockVaultResponse)
@katalepsis_monitor.instrument_api  # type: ignore
async def lock(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)) -> LockVaultResponse:  # type: ignore
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
    success = orch.lock()

    if success:
        return LockVaultResponse(
            status=Status.SUCCESS, message="Vault locked successfully."
        )
    else:
        logger.error("Failed to lock vault.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="A critical error occurred during vault locking.",
        )


@router_rag.get("/check_manifest", response_model=CheckManifestResponse)
@katalepsis_monitor.instrument_api  # type: ignore
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
    try:
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
    except Exception as e:
        logger.error("Failed to check manifest.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while checking the manifest: {str(e)}",
        )


@router_rag.post("/rebuild_index", response_model=RebuildIndexResponse)
@katalepsis_monitor.instrument_api  # type: ignore
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
    try:
        orch.rag.rebuild_index()
        return RebuildIndexResponse(
            status=Status.SUCCESS, message="Index rebuilt successfully."
        )
    except Exception as e:
        logger.error("Failed to rebuild index.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while rebuilding the index: {str(e)}",
        )


@router_rag.post("/scan_and_index", response_model=ScanAndIndexResponse)
@katalepsis_monitor.instrument_api  # type: ignore
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
    try:
        background_tasks.add_task(orch.rag.perform_initial_scan, [])
        return ScanAndIndexResponse(
            status=Status.SUCCESS,
            message="Scan and indexing started in the background.",
        )
    except Exception as e:
        logger.error("Failed to start scan and indexing.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while starting scan and indexing. ",
        )


@router_rag.post("/add_directories", response_model=DirectoriesAdditionResponse)
@katalepsis_monitor.instrument_api  # type: ignore
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
    try:
        background_tasks.add_task(orch.rag.add_watch_directories, request.directories)
        return DirectoriesAdditionResponse(
            status=Status.SUCCESS,
            message=f"Directories '{', '.join(request.directories)}' added for indexing.",
        )
    except Exception as e:
        logger.error("Failed to add directories.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while adding directories: {', '.join(request.directories)}",
        )


@router_rag.post("/remove_directories", response_model=DirectoriesRemovalResponse)
@katalepsis_monitor.instrument_api  # type: ignore
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
    try:
        for directory in request.directories:
            background_tasks.add_task(orch.rag.remove_watch_directories, [directory])
        return DirectoriesRemovalResponse(
            status=Status.SUCCESS,
            message=f"Directories '{', '.join(request.directories)}' removed from indexing.",
        )
    except Exception as e:
        logger.error("Failed to remove directories.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while removing directories: {', '.join(request.directories)}",
        )


app.include_router(router_vault)
app.include_router(router_chat)
app.include_router(router_rag)
