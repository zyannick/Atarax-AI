import pytest
import tempfile
from pathlib import Path
from unittest import mock
from ataraxai.praxis.utils.configs.llama_config_manager import LlamaConfigManager, LLAMA_CONFIG_FILENAME
import yaml

from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaConfig,
    LlamaModelParams,
    GenerationParams,
)

@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_initializes_default_config_if_not_exists(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    assert config_file.exists()
    assert isinstance(manager.config, LlamaConfig)

def test_loads_existing_config(temp_config_dir):
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    config_data = {
        "llama_cpp_model_params": {"model_path": "test-model"},
        "generation_params": {"n_predict": 123},
    }
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    manager = LlamaConfigManager(temp_config_dir)
    assert manager.config.llama_cpp_model_params.model_path == "test-model"
    assert manager.config.generation_params.n_predict == 123

def test_get_llm_params_and_generation_params(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    llm_params = manager.get_llama_cpp_params()
    gen_params = manager.get_generation_params()
    assert isinstance(llm_params, LlamaModelParams)
    assert isinstance(gen_params, GenerationParams)

def test_set_param_updates_and_saves(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager.set_param("llama_cpp_model_params", "model_path", "new-model")
    assert manager.config.llama_cpp_model_params.model_path == "new-model"
    manager.reload()
    assert manager.config.llama_cpp_model_params.model_path == "new-model"

def test_set_param_invalid_section_raises(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    with pytest.raises(ValueError):
        manager.set_param("invalid_section", "model_path", "value")

def test_set_param_invalid_key_raises(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    with pytest.raises(ValueError):
        manager.set_param("llama_cpp_model_params", "not_a_key", "value")

def test_reload_reloads_config(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager.set_param("llama_cpp_model_params", "model_path", "foo")
    manager.reload()
    assert manager.config.llama_cpp_model_params.model_path == "foo"

def test_load_or_initialize_handles_parse_error(temp_config_dir):
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    config_file.write_text("not: valid: yaml: [")
    with mock.patch("builtins.print"):
        manager = LlamaConfigManager(temp_config_dir)
    assert isinstance(manager.config, LlamaConfig)