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
from ataraxai.routes.rag_route.rag import router_rag
from ataraxai.routes.vault_route.vault import router_vault
from ataraxai.routes.chat_route.chat import router_chat
from ataraxai.routes.models_manager_route.models_manager import router_models_manager
from ataraxai.routes.configs_routes.user_preferences_route.user_preferences import router_user_preferences
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config import llama_cpp_router
from ataraxai.routes.configs_routes.rag_config_route.rag_config_route import rag_config_router
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
app.include_router(router_models_manager)
app.include_router(router_user_preferences)
app.include_router(llama_cpp_router)
app.include_router(rag_config_router)
