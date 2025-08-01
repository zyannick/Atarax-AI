from fastapi import APIRouter
from fastapi.params import Depends


from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo
from ataraxai.praxis.utils.configs.llama_config_manager import LlamaConfigManager
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

router_llama_cpp = APIRouter(
    prefix="/api/v1/llama_cpp_config", tags=["Llama CPP Config"]
)

def get_llama_config_manager(
    orch: AtaraxAIOrchestrator = Depends(get_unlocked_orchestrator) # type: ignore
) -> LlamaConfigManager:
    return orch.config_manager.llama_config_manager


@router_llama_cpp.get("/get_llama_cpp_config", response_model=LlamaCPPConfigResponse)
@handle_api_errors("Get Llama CPP Config", logger=logger)
async def get_llama_cpp_config(
    llama_config_manager: LlamaConfigManager = Depends(get_llama_config_manager),  # type: ignore
) -> LlamaCPPConfigResponse:
    """
    Endpoint to retrieve the current Llama CPP configuration.

    Args:
        llama_config_manager: The Llama config manager dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPConfigResponse: The current Llama CPP configuration.
    """
    config: LlamaModelParams = llama_config_manager.get_llama_cpp_params()
    if not config:
        return LlamaCPPConfigResponse(
            status=Status.FAILURE,
            message="Llama CPP configuration not found.",
            config=None,
        )
    return LlamaCPPConfigResponse(
        status=Status.SUCCESS,
        message="Llama CPP configuration retrieved successfully.",
        config=LlamaCPPConfigAPI(
            model_info=config.model_info,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            main_gpu=config.main_gpu,
            tensor_split=config.tensor_split,
            vocab_only=config.vocab_only,
            use_map=config.use_map,
            use_mlock=config.use_mlock,
        ),  # type:
    )


@router_llama_cpp.put("/update_llama_cpp_config", response_model=LlamaCPPConfigResponse)
@handle_api_errors("Update Llama CPP Config", logger=logger)
async def update_llama_cpp_config(
    config: LlamaCPPConfigAPI,
    llama_config_manager: LlamaConfigManager = Depends(get_llama_config_manager),  # type: ignore
) -> LlamaCPPConfigResponse:
    """
    Endpoint to update the Llama CPP configuration.

    Args:
        config: The new configuration to set.
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPConfigResponse: The updated Llama CPP configuration.
    """
    dict_config = config.model_dump()
    llama_model_params = LlamaModelParams(
        config_version=dict_config.get("config_version", "1.0"),
        model_info=LlamaCPPModelInfo(**(dict_config["model_info"])),
        n_ctx=dict_config["n_ctx"],
        n_gpu_layers=dict_config["n_gpu_layers"],
        main_gpu=dict_config["main_gpu"],
        tensor_split=dict_config["tensor_split"],
        vocab_only=dict_config["vocab_only"],
        use_map=dict_config["use_map"],
        use_mlock=dict_config["use_mlock"],
    )
    llama_config_manager.set_llama_cpp_params(llama_model_params)
    return LlamaCPPConfigResponse(
        status=Status.SUCCESS,
        message="Llama CPP configuration updated successfully.",
        config=LlamaCPPConfigAPI(**config.model_dump()),
    )


@router_llama_cpp.get(
    "/get_llama_cpp_generation_params", response_model=LlamaCPPGenerationParamsResponse
)
@handle_api_errors("Get Llama CPP Generation Params", logger=logger)
async def get_llama_cpp_generation_params(
    llama_config_manager: LlamaConfigManager = Depends(get_llama_config_manager),  # type: ignore
) -> LlamaCPPGenerationParamsResponse:
    """
    Endpoint to retrieve the current Llama CPP generation parameters.

    Args:
        orch: The unlocked orchestrator dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPGenerationParamsAPI: The current Llama CPP generation parameters.
    """
    generation_params = llama_config_manager.get_generation_params()
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


@router_llama_cpp.put(
    "/update_llama_cpp_generation_params",
    response_model=LlamaCPPGenerationParamsResponse,
)
@handle_api_errors("Update Llama CPP Generation Params", logger=logger)
async def update_llama_cpp_generation_params(
    generation_params: LlamaCPPGenerationParamsAPI,
    llama_config_manager: LlamaConfigManager = Depends(get_llama_config_manager),  # type: ignore
) -> LlamaCPPGenerationParamsResponse:
    """
    Endpoint to update the Llama CPP generation parameters.

    Args:
        generation_params: The new generation parameters to set.
        llama_config_manager: The Llama config manager dependency, injected via FastAPI's Depends.

    Returns:
        LlamaCPPGenerationParamsAPI: The updated Llama CPP generation parameters.
    """
    llama_config_manager.set_generation_params(
        GenerationParams(**generation_params.model_dump())
    )
    return LlamaCPPGenerationParamsResponse(
        status=Status.SUCCESS,
        message="Llama CPP generation parameters updated successfully.",
        params=LlamaCPPGenerationParamsAPI(**generation_params.model_dump()),
    )


