from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends
from typing import List
import uuid
import os
from ataraxai import __version__
from ataraxai.routes.chat_api_models import (
    CreateProjectRequest,
    ProjectResponse,
    CreateSessionRequest,
    SessionResponse,
    ChatMessageRequest,
    MessageResponse,
)
from ataraxai.routes.status import StatusResponse, Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator

from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator, katalepsis_monitor
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger


logger = AtaraxAILogger("ataraxai.praxis.chat")


router_chat = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router_chat.post("/projects", response_model=ProjectResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Create Project")
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

        return ProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
    except Exception as e:
        logger.error(f"Failed to create project {project_data.name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating the project: {project_data.name}",
        )


@router_chat.delete("/projects/{project_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Delete Project")
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    try:
        orch.chat.delete_project(project_id)
        return StatusResponse(
            status=Status.SUCCESS, message=f"Project {project_id} deleted successfully."
        )
    except Exception as e:
        logger.error(f"Failed to delete project {project_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the project: {project_id}",
        )


@router_chat.get("/projects/{project_id}", response_model=ProjectResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Get Project")
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    return ProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@router_chat.get("/projects", response_model=List[ProjectResponse])
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("List Projects")
async def list_projects(orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
    """
    Handles GET requests to "/v1/projects" and returns a list of projects.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        List[CreateProjectResponse]: A list of project details, each containing project_id, name, and description.

    Raises:
        Any exceptions raised by the orchestrator's list_projects method.

    Decorators:
        @app.get: Registers the endpoint with FastAPI.
        @get_katalepsis.instrument_api: Instruments the API for monitoring.
    """
    projects = orch.chat.list_projects()
    return [
        ProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
        for project in projects
    ]


@router_chat.get(
    "/projects/{project_id}/sessions", response_model=List[SessionResponse]
)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("List Sessions")
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
        get_katalepsis.instrument_api is used to monitor this API endpoint.
    """
    sessions = orch.chat.list_sessions(project_id)
    return [
        SessionResponse(
            session_id=session.id, title=session.title, project_id=session.project_id
        )
        for session in sessions
    ]


@router_chat.post("/session", response_model=SessionResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Create Session")
async def create_session(session_data: CreateSessionRequest, orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator)):  # type: ignore
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

        return SessionResponse(
            session_id=session.id, title=session.title, project_id=session.project_id
        )
    except Exception as e:
        logger.error(f"Failed to create session for project {session_data.project_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to create session. An error occurred: {str(e)}",
        )


@router_chat.delete("/sessions/{session_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Delete Session")
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    orch.chat.delete_session(session_id)
    return StatusResponse(
        status=Status.SUCCESS, message=f"Session {session_id} deleted successfully."
    )


@router_chat.get("/sessions/{session_id}", response_model=SessionResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Get Session")
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return SessionResponse(
        session_id=session.id, title=session.title, project_id=session.project_id
    )


@router_chat.post("/sessions/{session_id}/messages", response_model=MessageResponse)
@katalepsis_monitor.instrument_api  # type: ignore
@handle_api_errors("Send Message")
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    try:
        response = orch.chat.add_message(
            session_id=session_id, role="user", content=message_data.user_query
        )
        return MessageResponse(
            assistant_response=response.content,
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Failed to send message in session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while sending the message: {str(e)}",
        )
