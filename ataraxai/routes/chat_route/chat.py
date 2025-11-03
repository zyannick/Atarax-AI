import logging
import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status

from ataraxai.gateway.gateway_task_manager import GatewayTaskManager
from ataraxai.gateway.request_manager import RequestManager, RequestPriority
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.katalepsis import katalepsis_monitor
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.chat_route.chat_api_models import (
    CreateProjectRequestAPI,
    CreateSessionRequestAPI,
    MessageResponseAPI,
    ProjectResponseAPI,
    SessionResponseAPI,
    ListProjectsResponseAPI,
    ListSessionsResponseAPI,
    UpdateSessionRequestAPI,
)
from ataraxai.routes.dependency_api import (
    get_gatewaye_task_manager,
    get_logger,
    get_request_manager,
    get_unlocked_orchestrator,
)
from ataraxai.routes.status import StatusResponse
from ataraxai.routes.status import TaskStatus as Status

router_chat = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router_chat.post("/projects", response_model=ProjectResponseAPI)
@katalepsis_monitor.instrument_api("POST")
@handle_api_errors("Create Project")
async def create_new_project(
    project_data: CreateProjectRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    logger.info(
        f"Creating new project with name: {project_data.name} and description: {project_data.description}"
    )
    chat_manager = await orch.get_chat_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Create Project",
        func=chat_manager.create_project,
        name=project_data.name,
        description=project_data.description,
        priority=RequestPriority.HIGH,
    )
    project = await future
    return ProjectResponseAPI(
        status=Status.SUCCESS,
        project_id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )

@router_chat.put("/projects/{project_id}", response_model=ProjectResponseAPI)
@katalepsis_monitor.instrument_api("PUT")
@handle_api_errors("Update Project")
async def update_project(
    project_id: uuid.UUID,
    project_data: CreateProjectRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    logger.info(
        f"Updating project with ID: {project_id} to name: {project_data.name} and description: {project_data.description}"
    )
    chat_manager = await orch.get_chat_manager()
    existing_project = await chat_manager.get_project(project_id)
    if not existing_project:
         raise HTTPException(status_code=404, detail="Project not found")

    future = await req_manager.submit_request(
        request_name="Update Project",
        func=chat_manager.update_project,
        project_id=project_id,
        name=project_data.name,
        description=project_data.description,
        priority=RequestPriority.HIGH,
    )
    updated_project = await future 
    
    return ProjectResponseAPI(
        status=Status.SUCCESS,
        project_id=updated_project.id,
        name=updated_project.name,
        description=updated_project.description,
        created_at=updated_project.created_at,
        updated_at=updated_project.updated_at,
    )



@router_chat.delete("/projects/{project_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("DELETE")
@handle_api_errors("Delete Project")
async def delete_project(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    project = await chat_manager.get_project(project_id)
    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    future = await req_manager.submit_request(  # type: ignore
        request_name="Delete Project",
        func=chat_manager.delete_project,
        project_id=project_id,
        priority=RequestPriority.HIGH,
    )
    task_id = task_manager.create_task(future)  # type: ignore

    return StatusResponse(
        status=Status.PENDING,
        message=f"Project {project_id} deletion started.",
        task_id=task_id,
    )


@router_chat.get(("/projects/delete/{task_id}"), response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Project Deletion Status")
async def get_project_deletion_status(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    result = task_manager.get_task_status(task_id)
    if result is None:
        return StatusResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found.",
            task_id=task_id,
        )
    return StatusResponse(
        status=result.get("status", Status.ERROR),
        message="Project deletion status retrieved.",
        task_id=task_id,
    )


@router_chat.get("/projects/{project_id}", response_model=ProjectResponseAPI)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Project")
async def get_project(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Get Project",
        func=chat_manager.get_project,
        project_id=project_id,
        priority=RequestPriority.HIGH,
    )
    project = await future

    if not project:
        logger.error(f"Project with ID {project_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    return ProjectResponseAPI(
        status=Status.SUCCESS,
        project_id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router_chat.get("/projects", response_model=ListProjectsResponseAPI)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Projects")
async def list_projects(
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="List Projects",
        func=chat_manager.list_projects,
        priority=RequestPriority.HIGH,
    )
    projects = await future

    logger.info(f"Retrieved {len(projects)} projects.")
    list_projects : ListProjectsResponseAPI = ListProjectsResponseAPI(
        status=Status.SUCCESS,
        projects=[
            ProjectResponseAPI(
                status=Status.SUCCESS,
                project_id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
            for project in projects
        ],
    )
    return list_projects


@router_chat.get(
    "/projects/{project_id}/sessions", response_model=ListSessionsResponseAPI
)
@katalepsis_monitor.instrument_api("GET")  # type: ignore
@handle_api_errors("List Sessions")
async def list_sessions(
    project_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="List Sessions",
        func=chat_manager.list_sessions,
        project_id=project_id,
        priority=RequestPriority.HIGH,
    )
    sessions = await future
    list_sessions : ListSessionsResponseAPI = ListSessionsResponseAPI(
        status=Status.SUCCESS,
        sessions=[
            SessionResponseAPI(
                status=Status.SUCCESS,
                session_id=session.id,
                title=session.title,
                project_id=session.project_id,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            for session in sessions
        ],
    )
    return list_sessions


@router_chat.post("/sessions", response_model=SessionResponseAPI)
@katalepsis_monitor.instrument_api("POST")  # type: ignore
@handle_api_errors("Create Session")
async def create_session(
    session_data: CreateSessionRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    try:
        await chat_manager.get_project(session_data.project_id)
    except Exception as e:
        logger.error(f"Error retrieving project with ID {session_data.project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )

    future = await req_manager.submit_request(  # type: ignore
        request_name="Create Session",
        func=chat_manager.create_session,
        project_id=session_data.project_id,
        title=session_data.title,
        priority=RequestPriority.HIGH,
    )
    session = await future

    return SessionResponseAPI(
        status=Status.SUCCESS,
        session_id=session.id,
        title=session.title,
        project_id=session.project_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )

@router_chat.put("/sessions/{session_id}", response_model=SessionResponseAPI)
@katalepsis_monitor.instrument_api("PUT")
@handle_api_errors("Update Session")
async def update_session(
    session_id: uuid.UUID,
    session_data: UpdateSessionRequestAPI,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    future = await req_manager.submit_request(  # type: ignore
        request_name="Update Session",
        func=chat_manager.update_session,
        session_id=session_id,
        title=session_data.title,
        priority=RequestPriority.HIGH,
    )
    session = await future

    return SessionResponseAPI(
        status=Status.SUCCESS,
        session_id=session.id,
        title=session.title,
        project_id=session.project_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router_chat.delete("/sessions/{session_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("DELETE")
@handle_api_errors("Delete Session")
async def delete_session(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    future = await req_manager.submit_request(  # type: ignore
        request_name="Delete Session",
        func=chat_manager.delete_session,
        session_id=session_id,
        priority=RequestPriority.HIGH,
    )

    task_id = task_manager.create_task(future)  # type: ignore
    return StatusResponse(
        status=Status.PENDING,
        message=f"Session {session_id} deleted successfully.",
        task_id=task_id,
    )


@router_chat.get("/sessions/delete/{task_id}", response_model=StatusResponse)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Delete Session")
async def get_delete_session_status(
    task_id: str,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    result = task_manager.get_task_status(task_id)
    if result is None:
        logger.error(f"Task with ID {task_id} not found.")
        return StatusResponse(
            status=Status.ERROR,
            message=f"Task with ID {task_id} not found.",
            task_id=task_id,
        )
    return StatusResponse(
        status=result.get("status", Status.ERROR),
        message="Session deletion status retrieved.",
        task_id=task_id,
    )


@router_chat.get("/sessions/{session_id}", response_model=SessionResponseAPI)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Session")
async def get_session(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    return SessionResponseAPI(
        status=Status.SUCCESS,
        session_id=session.id,
        title=session.title,
        project_id=session.project_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router_chat.get(
    "/sessions/{session_id}/messages", response_model=List[MessageResponseAPI]
)
@katalepsis_monitor.instrument_api("GET")
@handle_api_errors("Get Messages")
async def get_messages(
    session_id: uuid.UUID,
    orch: Annotated[AtaraxAIOrchestrator, Depends(get_unlocked_orchestrator)],
    req_manager: Annotated[RequestManager, Depends(get_request_manager)],
    task_manager: Annotated[GatewayTaskManager, Depends(get_gatewaye_task_manager)],
    logger: Annotated[logging.Logger, Depends(get_logger)],
):
    chat_manager = await orch.get_chat_manager()
    session = await chat_manager.get_session(session_id)
    if not session:
        logger.error(f"Session with ID {session_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    messages = await chat_manager.get_messages_for_session(session_id=session_id)
    return [MessageResponseAPI.model_validate(msg) for msg in messages]
