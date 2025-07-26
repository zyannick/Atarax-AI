import pytest
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest import mock

from ataraxai.praxis.utils.configs.llama_config_manager import (
    LlamaConfigManager,
    LLAMA_CONFIG_FILENAME,
)
from ataraxai.praxis.utils.configs.config_schemas.llama_config_schema import (
    LlamaConfig,
    LlamaModelParams,
    GenerationParams,
)


@pytest.fixture
def temp_config_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_init_creates_default_config(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    assert config_file.exists()
    assert isinstance(manager.config, LlamaConfig)


def test_init_raises_on_file_path(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("dummy")
    with pytest.raises(ValueError):
        LlamaConfigManager(file_path)


def test_load_existing_config(temp_config_dir):
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    config_data = LlamaConfig().model_dump()
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    manager = LlamaConfigManager(temp_config_dir)
    assert isinstance(manager.config, LlamaConfig)


def test_save_and_reload(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager.config.llama_cpp_model_params.n_ctx = 1234
    manager._save()
    manager.reload()
    assert manager.config.llama_cpp_model_params.n_ctx == 1234


def test_get_and_set_llama_cpp_params(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    params = LlamaModelParams(n_ctx=2048)
    manager.set_llama_cpp_params(params)
    assert manager.get_llama_cpp_params().n_ctx == 2048


def test_get_and_set_generation_params(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    params = GenerationParams(n_predict=42)
    manager.set_generation_params(params)
    assert manager.get_generation_params().n_predict == 42


def test_set_param_valid(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager.set_param("llama_cpp_model_params", "n_ctx", 555)
    assert manager.config.llama_cpp_model_params.n_ctx == 555
    manager.set_param("generation_params", "n_predict", 77)
    assert manager.config.generation_params.n_predict == 77


def test_set_param_invalid_section(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    with pytest.raises(ValueError):
        manager.set_param("invalid_section", "n_ctx", 123)


def test_set_param_invalid_key(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    with pytest.raises(ValueError):
        manager.set_param("llama_cpp_model_params", "not_a_key", 123)


def test_save_creates_yaml(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager._save()
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    

def test_reload_loads_updated_config(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump({"llama_cpp_model_params": {"n_ctx": 9999}, "generation_params": {}}, f)
    manager.reload()
    assert manager.config.llama_cpp_model_params.n_ctx == 9999

def test_save_overwrites_existing_config(temp_config_dir):
    manager = LlamaConfigManager(temp_config_dir)
    manager.config.llama_cpp_model_params.n_ctx = 111
    manager._save()
    manager.config.llama_cpp_model_params.n_ctx = 222
    manager._save()
    manager.reload()
    assert manager.config.llama_cpp_model_params.n_ctx == 222

def test_logger_is_used_on_parse_error(temp_config_dir):
    config_file = temp_config_dir / LLAMA_CONFIG_FILENAME
    config_file.write_text("not: valid: yaml: [")
    mock_logger = mock.Mock()
    _ = LlamaConfigManager(temp_config_dir, logger=mock_logger)
    assert mock_logger.error.called

def test_config_path_parent_is_created(tmp_path):
    config_dir = tmp_path / "subdir1" / "subdir2"
    _ = LlamaConfigManager(config_dir)
    assert (config_dir / LLAMA_CONFIG_FILENAME).exists()
