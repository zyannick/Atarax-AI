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
    
    # @field_validator("local_path")
    # def validate_local_path(cls, value: str) -> str:
    #     if not value:
    #         raise ValueError("Local path must be a non-empty string.")
    #     if not Path(value).exists():
    #         raise ValueError("Local path must point to an existing file.")
    #     return value
    
    def to_hegemonikon(self) -> HegemonikonQuantizedModelInfo:
        return HegemonikonQuantizedModelInfo.from_dict(self.model_dump())


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
    
    def to_hegemonikon(self) -> HegemonikonBenchmarkMetrics:
        return HegemonikonBenchmarkMetrics.from_dict(self.model_dump())

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
        return HegemonikonBenchmarkParams.from_dict(self.model_dump())


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
    def __init__(self, quantized_model_info: QuantizedModelInfo, benchmark_params: BenchmarkParams, llama_model_params: LlamaModelParams):
        self.quantized_model_info = quantized_model_info
        self.benchmark_params = benchmark_params
        self.llama_model_params = llama_model_params
        self.benchmarker = HegemonikonLlamaBenchmarker(quantized_model_info.to_hegemonikon(), benchmark_params.to_hegemonikon(), llama_model_params.to_hegemonikon())



if __name__ == "__main__":
    import time

    quantized_model_info = QuantizedModelInfo(
        model_id="test-model",
        local_path="path/to/model.bin",
        last_modified="2023-10-01T12:00:00Z",
        quantisation_type="Q4_0",
        size_bytes=123456789,
    )

    benchmark_params = BenchmarkParams(
        n_gpu_layers=0,
        repetitions=1,
        warmup=True,
        generation_params=GenerationParams(
            n_predict=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repeat_penalty=1.1,
            penalty_last_n=64,
            penalty_freq=0.5,
            penalty_present=0.0,
            stop_sequences=["</s>"],
            n_batch=1,
            n_threads=4,
        ),
    )

    llama_model_params = LlamaModelParams(
        model_info=None,  # Assuming model_info is optional
        n_ctx=512,
        n_parts=-1,
        seed=-1,
        f16_kv=False,
        logits_all=False,
        vocab_only=False,
        use_mmap=True,
        use_mlock=False,
    )

    benchmark_runner = BenchmarkRunner(quantized_model_info, benchmark_params, llama_model_params)
    
    start_time = time.time()
    # result = benchmark_runner.run_benchmark()
    end_time = time.time()
    
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds.")
    # print(result)