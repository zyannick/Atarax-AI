from fastapi import APIRouter
from fastapi.params import Depends


from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
    LlamaCPPConfigAPI,
    LlamaCPPConfigResponse,
    LlamaCPPGenerationParamsAPI,
    LlamaCPPGenerationParamsResponse,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
)
from ataraxai.routes.status import Status
from ataraxai.praxis.ataraxai_orchestrator import AtaraxAIOrchestrator
from ataraxai.praxis.utils.ataraxai_logger import AtaraxAILogger
from ataraxai.praxis.utils.decorators import handle_api_errors
from ataraxai.routes.dependency_api import get_unlocked_orchestrator

logger = AtaraxAILogger("ataraxai.praxis.llama_cpp").get_logger()

llama_cpp_router = APIRouter(
    prefix="/api/v1/llama_cpp_config", tags=["Llama CPP Config"]
)


@llama_cpp_router.get("/get_llama_cpp_config", response_model=LlamaCPPConfigResponse)
@handle_api_errors("Get Llama CPP Config", logger=logger)
async def get_llama_cpp_config(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> LlamaCPPConfigResponse:
    """
    Endpoint to retrieve the current Llama CPP configuration.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPConfigResponse: The current Llama CPP configuration.
    """
    config = orch.config_manager.llama_config_manager.get_llama_cpp_params()
    if not config:
        return LlamaCPPConfigResponse(
            status=Status.FAILURE,
            message="Llama CPP configuration not found.",
            config=None,
        )
    return LlamaCPPConfigResponse(
        status=Status.SUCCESS,
        message="Llama CPP configuration retrieved successfully.",
        config=LlamaCPPConfigAPI(**config.model_dump()),
    )


@llama_cpp_router.put("/update_llama_cpp_config", response_model=LlamaCPPConfigResponse)
@handle_api_errors("Update Llama CPP Config", logger=logger)
async def update_llama_cpp_config(
    config: LlamaCPPConfigAPI,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> LlamaCPPConfigResponse:
    """
    Endpoint to update the Llama CPP configuration.

    Args:
        config: The new configuration to set.
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPConfigResponse: The updated Llama CPP configuration.
    """
    orch.config_manager.llama_config_manager.set_llama_cpp_params(
        LlamaModelParams(**config.model_dump())
    )
    return LlamaCPPConfigResponse(
        status=Status.SUCCESS,
        message="Llama CPP configuration updated successfully.",
        config=LlamaCPPConfigAPI(**config.model_dump()),
    )


@llama_cpp_router.get(
    "/get_llama_cpp_generation_params", response_model=LlamaCPPGenerationParamsResponse
)
@handle_api_errors("Get Llama CPP Generation Params", logger=logger)
async def get_llama_cpp_generation_params(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> LlamaCPPGenerationParamsResponse:
    """
    Endpoint to retrieve the current Llama CPP generation parameters.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPGenerationParamsAPI: The current Llama CPP generation parameters.
    """
    generation_params = orch.config_manager.llama_config_manager.get_generation_params()
    if not generation_params:
        return LlamaCPPGenerationParamsResponse(
            status=Status.FAILURE,
            message="Llama CPP generation parameters not found.",
            params=None,
        )
    return LlamaCPPGenerationParamsResponse(
        status=Status.SUCCESS,
        message="Llama CPP generation parameters retrieved successfully.",
        params=LlamaCPPGenerationParamsAPI(**generation_params.model_dump()),
    )


@llama_cpp_router.put(
    "/update_llama_cpp_generation_params",
    response_model=LlamaCPPGenerationParamsResponse,
)
@handle_api_errors("Update Llama CPP Generation Params", logger=logger)
async def update_llama_cpp_generation_params(
    generation_params: LlamaCPPGenerationParamsAPI,
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator),  # type: ignore
) -> LlamaCPPGenerationParamsResponse:
    """
    Endpoint to update the Llama CPP generation parameters.

    Args:
        generation_params: The new generation parameters to set.
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPGenerationParamsAPI: The updated Llama CPP generation parameters.
    """
    orch.config_manager.llama_config_manager.set_generation_params(
        GenerationParams(**generation_params.model_dump())
    )
    return LlamaCPPGenerationParamsResponse(
        status=Status.SUCCESS,
        message="Llama CPP generation parameters updated successfully.",
        params=LlamaCPPGenerationParamsAPI(**generation_params.model_dump()),
    )
