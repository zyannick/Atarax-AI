from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from fastapi import FastAPI, Form
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from fastapi import FastAPI, HTTPException, BackgroundTasks


class CreateProjectRequest(BaseModel):
    name: str
    description: str | None = None


class CreateSessionRequest(BaseModel):
    project_id: uuid.UUID
    title: str = Field(..., description="The initial title for the new chat session.")


class ChatMessageRequest(BaseModel):
    session_id: uuid.UUID
    user_query: str = Field(..., description="The user's message to the AI.")


class ChatMessageResponse(BaseModel):
    assistant_response: str = Field(..., description="The AI's generated response.")
    session_id: uuid.UUID = Field(..., description="The ID of the chat session.")


class CreateProjectResponse(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str


class CreateSessionResponse(BaseModel):
    session_id: uuid.UUID
    title: str
    project_id: uuid.UUID


app = FastAPI(
    title="AtaraxAI API",
    description="API for the AtaraxAI Local Assistant Engine",
    version="1.0.0",
)

orchestrator = AtaraxAIOrchestrator()


@app.post("/v1/projects", response_model=CreateProjectResponse)
async def create_new_project(project_data: CreateProjectRequest):
    try:
        project = orchestrator.create_project(
            name=project_data.name, description=project_data.description
        )

        return CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.delete("/v1/projects/{project_id}", response_model=CreateProjectResponse)
async def delete_project(project_id: uuid.UUID):
    project = orchestrator.db_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    orchestrator.delete_project(project_id)
    return CreateProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@app.get("/v1/projects/{project_id}", response_model=CreateProjectResponse)
async def get_project(project_id: uuid.UUID):
    project = orchestrator.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return CreateProjectResponse(
        project_id=project.id, name=project.name, description=project.description
    )


@app.get("/v1/projects", response_model=List[CreateProjectResponse])
async def list_projects():
    projects = orchestrator.list_projects()
    return [
        CreateProjectResponse(
            project_id=project.id, name=project.name, description=project.description
        )
        for project in projects
    ]


@app.get(
    "/v1/projects/{project_id}/sessions", response_model=List[CreateSessionResponse]
)
async def list_sessions(project_id: uuid.UUID):
    sessions = orchestrator.list_sessions(project_id)
    return [
        CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project.id
        )
        for session in sessions
    ]


@app.post("/v1/sessions", response_model=CreateSessionResponse)
async def create_new_session(
    session_data: CreateSessionRequest):
    try:
        session = orchestrator.create_session(
            project_id=session_data.project_id, title=session_data.title
        )

        return CreateSessionResponse(
            session_id=session.id, title=session.title, project_id=session.project.id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.delete("/v1/sessions/{session_id}", response_model=CreateSessionResponse)
async def delete_session(session_id: uuid.UUID):
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator.delete_session(session_id)
    return CreateSessionResponse(
        session_id=session.id, title=session.title, project_id=session.project.id
    )


@app.post("/v1/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def create_chat_message(
    session_id: uuid.UUID, message_data: ChatMessageRequest, background_tasks: BackgroundTasks
):
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        response = orchestrator.create_chat_message(
            session_id=session_id, user_query=message_data.user_query
        )
        background_tasks.add_task(orchestrator.process_chat_message, response)

        return ChatMessageResponse(
            assistant_response=response.assistant_response, 
            session_id=response.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    print("API is shutting down. Closing orchestrator resources.")
    orchestrator.shutdown()
