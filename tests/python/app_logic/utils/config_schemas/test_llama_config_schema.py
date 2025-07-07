import pytest

from ataraxai.app_logic.utils.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
    LlamaConfig,
)

def test_llama_model_params_defaults():
    params = LlamaModelParams()
    assert params.config_version == 1.0
    assert params.model_path == "data/last_models/models/llama/Qwen3-30B-A3B-UD-IQ1_M.gguf"
    assert params.n_ctx == 2048
    assert params.n_gpu_layers == 0
    assert params.main_gpu == 0
    assert params.tensor_split is False
    assert params.vocab_only is False
    assert params.use_map is False
    assert params.use_mlock is False

def test_llama_model_params_custom_values():
    params = LlamaModelParams(
        config_version=2.0,
        model_path="model.bin",
        n_ctx=4096,
        n_gpu_layers=2,
        main_gpu=1,
        tensor_split=True,
        vocab_only=True,
        use_map=True,
        use_mlock=True,
    )
    assert params.config_version == 2.0
    assert params.model_path == "model.bin"
    assert params.n_ctx == 4096
    assert params.n_gpu_layers == 2
    assert params.main_gpu == 1
    assert params.tensor_split is True
    assert params.vocab_only is True
    assert params.use_map is True
    assert params.use_mlock is True

def test_generation_params_defaults():
    params = GenerationParams()
    assert params.config_version == 1.0
    assert params.n_predict == 128
    assert params.temp == 0.8
    assert params.top_k == 40
    assert params.top_p == 0.95
    assert params.repeat_penalty == 1.2
    assert params.penalty_last_n == 64
    assert params.penalty_freq == 0.7
    assert params.penalty_present == 0.0
    assert params.stop_sequences == ["</s>", "\n\n", "User:"]
    assert params.n_batch == 1
    assert params.n_threads == 4

def test_generation_params_custom_values():
    params = GenerationParams(
        config_version=2.0,
        n_predict=256,
        temp=0.5,
        top_k=10,
        top_p=0.8,
        repeat_penalty=1.0,
        penalty_last_n=32,
        penalty_freq=0.5,
        penalty_present=0.2,
        stop_sequences=["<end>", "STOP"],
        n_batch=2,
        n_threads=8,
    )
    assert params.config_version == 2.0
    assert params.n_predict == 256
    assert params.temp == 0.5
    assert params.top_k == 10
    assert params.top_p == 0.8
    assert params.repeat_penalty == 1.0
    assert params.penalty_last_n == 32
    assert params.penalty_freq == 0.5
    assert params.penalty_present == 0.2
    assert params.stop_sequences == ["<end>", "STOP"]
    assert params.n_batch == 2
    assert params.n_threads == 8

def test_llama_config_defaults():
    config = LlamaConfig()
    assert config.config_version == 1.0
    assert isinstance(config.llama_cpp_model_params, LlamaModelParams)
    assert isinstance(config.generation_params, GenerationParams)
    assert config.llama_cpp_model_params.n_ctx == 2048
    assert config.generation_params.n_predict == 128

def test_llama_config_custom_values():
    model_params = LlamaModelParams(model_path="foo.bin", n_ctx=1024)
    gen_params = GenerationParams(n_predict=10, temp=0.1)
    config = LlamaConfig(
        config_version=2.0,
        llama_cpp_model_params=model_params,
        generation_params=gen_params,
    )
    assert config.config_version == 2.0
    assert config.llama_cpp_model_params.model_path == "foo.bin"
    assert config.llama_cpp_model_params.n_ctx == 1024
    assert config.generation_params.n_predict == 10
    assert config.generation_params.temp == 0.1
    
    
def test_llama_model_params_type_enforcement():
    with pytest.raises(ValueError):
        LlamaModelParams(n_ctx="not_an_int")
    with pytest.raises(ValueError):
        LlamaModelParams(config_version="not_a_float")
    with pytest.raises(ValueError):
        LlamaModelParams(tensor_split="not_a_bool")

def test_generation_params_type_enforcement():
    with pytest.raises(ValueError):
        GenerationParams(n_predict="not_an_int")
    with pytest.raises(ValueError):
        GenerationParams(temp="not_a_float")
    with pytest.raises(ValueError):
        GenerationParams(stop_sequences="not_a_list")

def test_llama_config_type_enforcement():
    with pytest.raises(ValueError):
        LlamaConfig(config_version="not_a_float")
    with pytest.raises(ValueError):
        LlamaConfig(llama_cpp_model_params="not_a_llama_model_params")
    with pytest.raises(ValueError):
        LlamaConfig(generation_params="not_a_generation_params")

def test_llama_config_nested_dict_instantiation():
    config = LlamaConfig(
        llama_cpp_model_params={"model_path": "bar.bin", "n_ctx": 512},
        generation_params={"n_predict": 5, "temp": 0.2},
    )
    assert isinstance(config.llama_cpp_model_params, LlamaModelParams)
    assert config.llama_cpp_model_params.model_path == "bar.bin"
    assert config.llama_cpp_model_params.n_ctx == 512
    assert isinstance(config.generation_params, GenerationParams)
    assert config.generation_params.n_predict == 5
    assert config.generation_params.temp == 0.2

def test_llama_config_dict_serialization():
    config = LlamaConfig()
    config_dict = config.model_dump()
    print(config_dict)  # For debugging purposes
    assert "llama_cpp_model_params" in config_dict
    assert "generation_params" in config_dict
    assert config_dict["llama_cpp_model_params"]["n_ctx"] == 2048
    assert config_dict["generation_params"]["n_predict"] == 128
