from fastapi import  FastAPI
from contextlib import asynccontextmanager
from fastapi.params import Depends
import os
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestratorFactory
from ataraxai import __version__
from ataraxai.routes.status import StatusResponse, Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from ataraxai.praxis.katalepsis import Katalepsis
from ataraxai.routes.rag_api.rag import router_rag
from ataraxai.routes.vault_api.vault import router_vault
from ataraxai.routes.chat_api.chat import router_chat
from ataraxai.routes.dependency_api import get_orchestrator

os.environ.setdefault("ENVIRONMENT", "development")  # Default to development if not set
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


logger = AtaraxAILogger("ataraxai.praxis.api").get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orchestrator = AtaraxAIOrchestratorFactory.create_orchestrator()
    app.state.katalepsis_monitor = Katalepsis()
    yield
    logger.info("API is shutting down. Closing orchestrator resources.")
    app.state.orchestrator.shutdown()


if ENVIRONMENT == "development":
    app = FastAPI(
        title="AtaraxAI API",
        description="API for the AtaraxAI Local Assistant Engine",
        version=__version__,
        lifespan=lifespan,
    )
else:
    app = FastAPI(
        title="AtaraxAI API",
        description="API for the AtaraxAI Local Assistant Engine",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "test"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://test"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/status", response_model=StatusResponse)
async def get_state(orch: AtaraxAIOrchestrator = Depends(get_orchestrator)) -> StatusResponse:  # type: ignore
    return StatusResponse(
        status=Status.SUCCESS,
        message=f"AtaraxAI is currently in state: {orch.state.name}",
    )


@app.get("/v1/health", response_model=StatusResponse)
async def get_health(orch: AtaraxAIOrchestrator = Depends(get_orchestrator)) -> StatusResponse:  # type: ignore
    return StatusResponse(
        status=Status.SUCCESS,
        message="AtaraxAI is healthy.",
    )


app.include_router(router_vault)
app.include_router(router_chat)
app.include_router(router_rag)
