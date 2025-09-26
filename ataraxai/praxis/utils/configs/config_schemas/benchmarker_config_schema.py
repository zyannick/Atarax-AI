from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from ataraxai.hegemonikon_py import HegemonikonBenchmarkMetrics  # type: ignore
from ataraxai.hegemonikon_py import HegemonikonBenchmarkParams  # type: ignore
from ataraxai.hegemonikon_py import (
    HegemonikonQuantizedModelInfo,  # type: ignore; type: ignore
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    GenerationParams,
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

    def to_hegemonikon(self) -> HegemonikonQuantizedModelInfo:
        # cpp_obj: HegemonikonQuantizedModelInfo = HegemonikonQuantizedModelInfo()  # type: ignore
        # cpp_obj.model_id = self.model_id
        # cpp_obj.local_path = self.local_path
        # cpp_obj.last_modified = self.last_modified or ""
        # cpp_obj.quantization = self.quantisation_type
        # cpp_obj.fileSize = self.size_bytes
        # return cpp_obj  # type: ignore
        return HegemonikonQuantizedModelInfo.from_dict(self.model_dump())  # type: ignore


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
        default=[],
        description="List of individual generation times in milliseconds.",
    )
    tokens_per_second_history: List[float] = Field(
        default=[],
        description="List of tokens per second recorded at different intervals.",
    )
    ttft_history_ms: List[float] = Field(
        default=[],
        description="List of time to first token measurements in milliseconds.",
    )
    end_to_end_latency_history_ms: List[float] = Field(
        default=[],
        description="List of end-to-end latency measurements in milliseconds.",
    )
    decode_times_history_ms: List[float] = Field(
        default=[], description="List of decode times in milliseconds."
    )

    avg_ttft_ms: float = Field(
        default=0.0, description="Average time to first token in milliseconds."
    )
    avg_decode_time_ms: float = Field(
        default=0.0, description="Average decode time in milliseconds."
    )
    avg_end_to_end_time_latency_ms: float = Field(
        default=0.0, description="Average end-to-end latency in milliseconds."
    )

    p50_latency_ms: float = Field(
        default=0.0, description="50th percentile latency in milliseconds."
    )
    p95_latency_ms: float = Field(
        default=0.0, description="95th percentile latency in milliseconds."
    )
    p99_latency_ms: float = Field(
        default=0.0, description="99th percentile latency in milliseconds."
    )

    @field_validator(
        "load_time_ms", "generation_time_ms", "total_time_ms", "memory_usage_mb"
    )
    def validate_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        return value

    @classmethod
    def from_hegemonikon(
        cls, cpp_obj: HegemonikonBenchmarkMetrics
    ) -> "BenchmarkMetrics":
        data = {
            "load_time_ms": cpp_obj.load_time_ms,
            "generation_time_ms": cpp_obj.generation_time_ms,
            "total_time_ms": cpp_obj.total_time_ms,
            "tokens_generated": cpp_obj.tokens_generated,
            "token_per_second": cpp_obj.tokens_per_second,
            "error_message": cpp_obj.errorMessage,
            "memory_usage_mb": cpp_obj.memory_usage_mb,
            "success": cpp_obj.success,
            "generation_time_history_ms": cpp_obj.generation_time_history_ms,
            "tokens_per_second_history": cpp_obj.tokens_per_second_history,
            "ttft_history_ms": cpp_obj.ttft_history_ms,
            "end_to_end_latency_history_ms": cpp_obj.end_to_end_latency_history_ms,
            "decode_times_history_ms": cpp_obj.decode_times_history_ms,
            "avg_ttft_ms": cpp_obj.avg_ttft_ms,
            "avg_decode_time_ms": cpp_obj.avg_decode_time_ms,
            "avg_end_to_end_time_latency_ms": cpp_obj.avg_end_to_end_time_latency_ms,
            "p50_latency_ms": cpp_obj.p50_latency_ms,
            "p95_latency_ms": cpp_obj.p95_latency_ms,
            "p99_latency_ms": cpp_obj.p99_latency_ms,
        }
        return cls(**data)

    def to_hegemonikon(self) -> HegemonikonBenchmarkMetrics:
        # cpp_obj: HegemonikonBenchmarkMetrics = HegemonikonBenchmarkMetrics()  # type: ignore
        # cpp_obj.load_time_ms = self.load_time_ms
        # cpp_obj.generation_time_ms = self.generation_time_ms
        # cpp_obj.total_time_ms = self.total_time_ms
        # cpp_obj.tokens_generated = self.tokens_generated
        # cpp_obj.tokens_per_second = self.token_per_second
        # cpp_obj.memory_usage_mb = self.memory_usage_mb
        # cpp_obj.success = self.success
        # cpp_obj.errorMessage = self.error_message or ""
        # cpp_obj.generation_time_history_ms = self.generation_time_history_ms
        # cpp_obj.tokens_per_second_history = self.tokens_per_second_history
        # cpp_obj.ttft_history_ms = self.ttft_history_ms
        # cpp_obj.end_to_end_latency_history_ms = self.end_to_end_latency_history_ms
        # cpp_obj.decode_times_history_ms = self.decode_times_history_ms
        # cpp_obj.avg_ttft_ms = self.avg_ttft_ms
        # cpp_obj.avg_decode_time_ms = self.avg_decode_time_ms
        # cpp_obj.avg_end_to_end_time_latency_ms = self.avg_end_to_end_time_latency_ms
        # cpp_obj.p50_latency_ms = self.p50_latency_ms
        # cpp_obj.p95_latency_ms = self.p95_latency_ms
        # cpp_obj.p99_latency_ms = self.p99_latency_ms
        # return cpp_obj  # type: ignore
        return HegemonikonBenchmarkMetrics.from_dict(self.model_dump())  # type: ignore

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkMetrics":
        return cls(**data)


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

    def to_hegemonikon(self) -> HegemonikonBenchmarkParams:
        # cpp_obj: HegemonikonBenchmarkParams = HegemonikonBenchmarkParams()  # type: ignore
        # cpp_obj.n_gpu_layers = self.n_gpu_layers
        # cpp_obj.repetitions = self.repetitions
        # cpp_obj.warmup = self.warmup
        # cpp_obj.generation_params = self.generation_params.to_hegemonikon() # type: ignore
        # return cpp_obj  # type: ignore
        return HegemonikonBenchmarkParams.from_dict(self.model_dump())  # type: ignore


class BenchmarkResult(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model.")
    metrics: BenchmarkMetrics = Field(
        ..., description="Benchmark metrics for the model."
    )

    @classmethod
    def from_hegemonikon(
        cls, cpp_obj: HegemonikonBenchmarkMetrics
    ) -> "BenchmarkResult":
        return cls(
            model_id=cpp_obj.model_id,  # type: ignore
            metrics=BenchmarkMetrics.from_hegemonikon(cpp_obj.metrics),  # type: ignore
        )

    @field_validator("model_id")
    def validate_model_id(cls, value: str) -> str:
        if not value:
            raise ValueError("Model ID must be a non-empty string.")
        return value
