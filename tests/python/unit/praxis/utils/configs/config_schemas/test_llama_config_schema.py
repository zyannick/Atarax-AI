import pytest
from ataraxai.praxis.modules.models_manager.models_manager import LlamaCPPModelInfo

from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaModelParams,
    GenerationParams,
    LlamaConfig,
)

def test_llama_model_params_defaults():
    params = LlamaModelParams()
    assert params.config_version == "1.0"
    assert params.n_ctx == 2048
    assert params.n_gpu_layers == 0
    assert params.main_gpu == 0
    assert params.tensor_split is False
    assert params.vocab_only is False
    assert params.use_map is False
    assert params.use_mlock is False
    assert params.model_path == ""
    assert not params.is_setup_complete()

def test_llama_model_params_with_model_info():
    model_info = LlamaCPPModelInfo(organization="test_org", repo_id="test_repo", filename="test_model.bin", local_path="/models/test_model.bin", metadata={})
    params = LlamaModelParams(model_info=model_info)
    assert params.model_path == "/models/test_model.bin"
    assert params.is_setup_complete()

def test_generation_params_defaults():
    params = GenerationParams()
    assert params.config_version == "1.0"
    assert params.n_predict == 128
    assert params.temperature == 0.8
    assert params.top_k == 40
    assert params.top_p == 0.95
    assert params.repeat_penalty == 1.2
    assert params.penalty_last_n == 64
    assert params.penalty_freq == 0.7
    assert params.penalty_present == 0.0
    assert params.stop_sequences == ["</s>", "\n\n", "User:"]
    assert params.n_batch == 1
    assert params.n_threads == 4
    assert params.is_setup_complete()

@pytest.mark.parametrize("field,value", [
    ("temperature", -0.1),
    ("top_k", -1),
    ("top_p", -0.5),
])
def test_generation_params_negative_values(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        GenerationParams(**kwargs)

def test_llama_config_defaults():
    config = LlamaConfig()
    assert config.config_version == "1.0"
    assert isinstance(config.llama_cpp_model_params, LlamaModelParams)
    assert isinstance(config.generation_params, GenerationParams)
    assert not config.is_setup_complete()

def test_llama_config_complete_setup():
    model_info = LlamaCPPModelInfo(organization="test_org", repo_id="test_repo", filename="test_model.bin", local_path="/models/test_model.bin", metadata={})
    llama_params = LlamaModelParams(model_info=model_info)
    gen_params = GenerationParams(n_predict=10, temperature=0.5, top_k=5, top_p=0.9)
    config = LlamaConfig(llama_cpp_model_params=llama_params, generation_params=gen_params)
    assert config.is_setup_complete()