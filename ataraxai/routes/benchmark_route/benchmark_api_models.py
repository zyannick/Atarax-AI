from datetime import datetime
from typing import Any, Dict,Optional

from pydantic import BaseModel, Field

from ataraxai.praxis.modules.benchmark.benchmark_queue_manager import BenchmarkJobStatus
from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    BenchmarkResult,
)
from ataraxai.routes.configs_routes.llama_cpp_config_route.llama_cpp_config_api_models import (
    LlamaCPPConfigAPI,
    LlamaCPPGenerationParamsAPI,
)

class QuantizedModelInfoAPI(BaseModel):
    model_id: str = Field(..., description="ID of the quantized model.")
    local_path: str = Field(..., description="Local path to the quantized model file.")
    last_modified: str = Field(..., description="Timestamp of the last modification.")
    quantisation_type: str = Field(..., description="Quantization type of the model.")
    size_bytes: int = Field(..., description="Size of the model file in bytes.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizedModelInfoAPI":
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BenchmarkParamsAPI(BaseModel):
    n_gpu_layers: int = Field(..., description="Number of GPU layers to use.")
    repetitions: int = Field(
        ..., description="Number of repetitions for the benchmark."
    )
    warmup: bool = Field(..., description="Whether to perform warmup runs.")
    generation_params: LlamaCPPGenerationParamsAPI = Field(
        ..., description="Parameters for text generation."
    )


class BenchmarkJobAPI(BaseModel):
    id: str = Field(..., description="Unique identifier for the benchmark job.")
    model_info: QuantizedModelInfoAPI = Field(
        ..., description="Information about the quantized model to benchmark."
    )
    benchmark_params: BenchmarkParamsAPI = Field(
        ..., description="Parameters for the benchmark."
    )
    llama_model_params: LlamaCPPConfigAPI = Field(
        ..., description="Llama model parameters to use during benchmarking."
    )
    status: BenchmarkJobStatus = Field(
        default=BenchmarkJobStatus.QUEUED,
        description="Current status of the benchmark job.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when the job was created.",
    )
    started_at: Optional[str] = Field(
        default=None, description="Timestamp when the job started."
    )
    completed_at: Optional[str] = Field(
        default=None, description="Timestamp when the job completed."
    )
    result: Optional[BenchmarkResult] = Field(
        default=None, description="Result of the benchmark job."
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if the job failed."
    )
