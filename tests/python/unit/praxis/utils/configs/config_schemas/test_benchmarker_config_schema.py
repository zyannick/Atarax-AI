import pytest
from pathlib import Path
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import GenerationParams

from ataraxai.praxis.utils.configs.config_schemas.benchmarker_config_schema import (
    QuantizedModelInfo,
    BenchmarkMetrics,
    BenchmarkParams,
    BenchmarkResult,
)

def test_quantized_model_info_valid(tmp_path : Path):
    file_path = tmp_path / "model.bin"
    file_path.write_text("dummy")
    info = QuantizedModelInfo(
        model_id="test-model",
        local_path=str(file_path),
        last_modified="2024-06-01T12:00:00",
        quantisation_type="int8",
        size_bytes=123456,
    )
    assert info.model_id == "test-model"
    assert info.size_bytes == 123456

def test_quantized_model_info_invalid_size_bytes(tmp_path : Path):
    file_path = tmp_path / "model.bin"
    file_path.write_text("dummy")
    with pytest.raises(ValueError):
        QuantizedModelInfo(
            model_id="test-model",
            local_path=str(file_path),
            last_modified="2024-06-01T12:00:00",
            quantisation_type="int8",
            size_bytes=-1,
        )

def test_quantized_model_info_invalid_local_path():
    with pytest.raises(ValueError):
        QuantizedModelInfo(
            model_id="test-model",
            local_path="nonexistent/path/model.bin",
            last_modified="2024-06-01T12:00:00",
            quantisation_type="int8",
            size_bytes=123,
        )

def test_benchmark_metrics_non_negative():
    metrics = BenchmarkMetrics(
        load_time_ms=10.0,
        generation_time_ms=20.0,
        total_time_ms=30.0,
        tokens_generated=100,
        token_per_second=5.0,
        error_message="",
        memory_usage_mb=256.0,
        success=True,
    )
    assert metrics.load_time_ms == 10.0
    assert metrics.success is True

@pytest.mark.parametrize("field,value", [
    ("load_time_ms", -1.0),
    ("generation_time_ms", -1.0),
    ("total_time_ms", -1.0),
    ("memory_usage_mb", -1.0),
])
def test_benchmark_metrics_negative_values(field: str, value: float):
    kwargs = dict(
        load_time_ms=1.0,
        generation_time_ms=1.0,
        total_time_ms=1.0,
        tokens_generated=1,
        token_per_second=1.0,
        error_message="",
        memory_usage_mb=1.0,
        success=True,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        BenchmarkMetrics(**kwargs)

def test_benchmark_metrics_from_dict():
    data = {
        "load_time_ms": 1.0,
        "generation_time_ms": 2.0,
        "total_time_ms": 3.0,
        "tokens_generated": 10,
        "token_per_second": 2.5,
        "error_message": "",
        "memory_usage_mb": 128.0,
        "success": True,
    }
    metrics = BenchmarkMetrics.from_dict(data)
    assert metrics.tokens_generated == 10

def test_benchmark_params_valid():
    gen_params = GenerationParams(
        temperature=0.7,
        top_p=0.9,
        stop_sequences=[],
    )
    params = BenchmarkParams(
        n_gpu_layers=2,
        repetitions=3,
        warmup=True,
        generation_params=gen_params,
    )
    assert params.n_gpu_layers == 2
    assert params.repetitions == 3

@pytest.mark.parametrize("field,value", [
    ("n_gpu_layers", -1),
    ("repetitions", -1),
])
def test_benchmark_params_negative(field, value):
    gen_params = GenerationParams(
        temperature=0.7,
        top_p=0.9,
        stop_sequences=[],
    )
    kwargs = dict(
        n_gpu_layers=1,
        repetitions=1,
        warmup=True,
        generation_params=gen_params,
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        BenchmarkParams(**kwargs)

def test_benchmark_result_valid():
    metrics = BenchmarkMetrics(
        load_time_ms=1.0,
        generation_time_ms=2.0,
        total_time_ms=3.0,
        tokens_generated=10,
        token_per_second=2.5,
        error_message="",
        memory_usage_mb=128.0,
        success=True,
    )
    result = BenchmarkResult(
        model_id="test-model",
        metrics=metrics,
    )
    assert result.model_id == "test-model"

def test_benchmark_result_invalid_model_id():
    metrics = BenchmarkMetrics(
        load_time_ms=1.0,
        generation_time_ms=2.0,
        total_time_ms=3.0,
        tokens_generated=10,
        token_per_second=2.5,
        error_message="",
        memory_usage_mb=128.0,
        success=True,
    )
    with pytest.raises(ValueError):
        BenchmarkResult(
            model_id="",
            metrics=metrics,
        )