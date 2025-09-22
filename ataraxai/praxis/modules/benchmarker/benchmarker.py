from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field, field_validator

from ataraxai.hegemonikon_py import (  # type: ignore
    HegemonikonBenchmarkMetrics,
    HegemonikonBenchmarkParams,
    HegemonikonBenchmarkResult,
    HegemonikonLlamaModelParams,
    HegemonikonQuantizedModelInfo,
    HegemonikonLlamaBenchmarker,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    GenerationParams,
    LlamaModelParams,
)


class QuantizedModelInfo(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model.")
    local_path: str = Field(..., description="Local filesystem path to the model.")
    last_modified: str = Field(..., description="Timestamp of the last modification.")
    quantisation_type: str = Field(
        ..., description="Type of quantization applied to the model."
    )
    size_bytes: int = Field(..., description="Size of the model file in bytes.")

    @field_validator("size_bytes")
    def validate_size_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Size in bytes must be non-negative.")
        return value
    
    @field_validator("local_path")
    def validate_local_path(cls, value: str) -> str:
        if not value:
            raise ValueError("Local path must be a non-empty string.")
        if not Path(value).exists():
            raise ValueError("Local path must point to an existing file.")
        return value


class BenchmarkMetrics(BaseModel):
    load_time_ms: float = Field(
        ..., description="Time taken to load the model in milliseconds."
    )
    generation_time_ms: float = Field(
        ..., description="Time taken for generation in milliseconds."
    )
    total_time_ms: float = Field(
        ..., description="Total time taken for the benchmark in milliseconds."
    )
    tokens_generated: int = Field(
        ..., description="Number of tokens generated during the benchmark."
    )
    token_per_second: float = Field(..., description="Tokens generated per second.")
    error_message: str = Field(
        "", description="Error message if any error occurred during benchmarking."
    )
    memory_usage_mb: float = Field(
        ..., description="Memory usage in megabytes during the benchmark."
    )
    success: bool = Field(..., description="Indicates if the benchmark was successful.")

    generation_time_history_ms: List[float] = Field(
        default_factory=list,
        description="List of individual generation times in milliseconds.",
    )
    token_per_second_times_history_ms: List[float] = Field(
        default_factory=list,
        description="List of tokens per second recorded at different intervals.",
    )
    ttft_history_ms: List[float] = Field(
        default_factory=list,
        description="List of time to first token measurements in milliseconds.",
    )
    end_to_end_latency_history_ms: List[float] = Field(
        default_factory=list,
        description="List of end-to-end latency measurements in milliseconds.",
    )
    decode_times_ms: List[float] = Field(
        default_factory=list, description="List of decode times in milliseconds."
    )

    avg_ttft_ms: float = Field(
        0.0, description="Average time to first token in milliseconds."
    )
    avg_decode_time_ms: float = Field(
        0.0, description="Average decode time in milliseconds."
    )
    avg_end_to_end_time_latency_ms: float = Field(
        0.0, description="Average end-to-end latency in milliseconds."
    )

    p50_latency_ms: float = Field(
        0.0, description="50th percentile latency in milliseconds."
    )
    p95_latency_ms: float = Field(
        0.0, description="95th percentile latency in milliseconds."
    )
    p99_latency_ms: float = Field(
        0.0, description="99th percentile latency in milliseconds."
    )

    @field_validator("load_time_ms", "generation_time_ms", "total_time_ms", "memory_usage_mb")
    def validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        return value
    

class BenchmarkParams(BaseModel):
    n_gpu_layers: int = Field(..., description="Number of GPU layers to use.")
    repetitions: int = Field(
        ..., description="Number of repetitions for the benchmark."
    )
    warmup: bool = Field(..., description="Whether to perform warmup runs.")
    generation_params: GenerationParams = Field(
        ..., description="Parameters for text generation."
    )

    @field_validator("n_gpu_layers", "repetitions")
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be a non-negative integer.")
        return value


class BenchmarkResult(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model.")
    metrics: BenchmarkMetrics = Field(
        ..., description="Benchmark metrics for the model."
    )
    benchmark_params: BenchmarkParams = Field(
        ..., description="Parameters used for benchmarking."
    )
    llama_model_params: LlamaModelParams = Field(
        ..., description="Llama model parameters used during benchmarking."
    )

    @field_validator("model_id")
    def validate_model_id(cls, value: str) -> str:
        if not value:
            raise ValueError("Model ID must be a non-empty string.")
        return value


class BenchmarkRunner:
    def __init__(self, model_info, benchmark_params, llama_model_params):
        self.model_info = model_info
        self.benchmark_params = benchmark_params
        self.llama_model_params = llama_model_params

    def run(self):
        pass
