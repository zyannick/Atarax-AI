from fastapi import APIRouter
from fastapi.params import Depends
from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from ataraxai.routes.configs_routes.rag_config_route.rag_config_api_models import RagConfigAPI, RagConfigResponse
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator



logger = AtaraxAILogger("ataraxai.praxis.rag_config").get_logger()

rag_config_router = APIRouter(
    prefix="/api/v1/rag_config", tags=["RAG Config"]
)

@rag_config_router.get("/get_rag_config", response_model=RagConfigResponse)
@handle_api_errors("Get RAG Config", logger=logger)
async def get_rag_config(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> RagConfigResponse:
    """
    Endpoint to retrieve the current RAG configuration.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        RagConfigResponse: The current RAG configuration.
    """
    config = orch.config_manager.rag_config_manager.get_config()
    if not config:
        return RagConfigResponse(
            status=Status.FAILURE,
            message="RAG configuration not found.",
            config=None,
        )
    return RagConfigResponse(
        status=Status.SUCCESS,
        message="RAG configuration retrieved successfully.",
        config=RagConfigAPI(**config.model_dump()),
    )
    
    
@rag_config_router.put("/update_rag_config", response_model=RagConfigResponse)
@handle_api_errors("Update RAG Config", logger=logger)
async def update_rag_config(
    config: RagConfigAPI,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> RagConfigResponse:
    """
    Endpoint to update the RAG configuration.

    Args:
        config (RagConfigAPI): The new RAG configuration to set.
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        RagConfigResponse: The updated RAG configuration.
    """
    rag_config = orch.config_manager.rag_config_manager
    rag_config.update_config(RAGConfig(**config.model_dump()))
    return RagConfigResponse(
        status=Status.SUCCESS,
        message="RAG configuration updated successfully.",
        config=RagConfigAPI(**rag_config.get_config().model_dump()),
    )