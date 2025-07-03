import pytest
import yaml
from pathlib import Path
from unittest import mock

from ataraxai.app_logic.utils.whisper_config_manager import (
    WhisperConfigManager,
    WHISPER_CONFIG_FILENAME,
)
from ataraxai.app_logic.utils.config_schemas.whisper_config_schema import (
    WhisperConfig,
    WhisperModelParams,
    WhisperTranscriptionParams,
)


@pytest.fixture
def tmp_config_dir(tmp_path):
    return tmp_path


def test_initializes_default_config_when_file_missing(tmp_config_dir):
    manager = WhisperConfigManager(tmp_config_dir)
    config_file = tmp_config_dir / WHISPER_CONFIG_FILENAME
    assert config_file.exists()
    config = manager.get_config()
    assert isinstance(config, WhisperConfig)
    assert isinstance(config.whisper_model_params, WhisperModelParams)
    assert isinstance(config.whisper_transcription_params, WhisperTranscriptionParams)


def test_loads_existing_config(tmp_config_dir):
    config_data = {
        "whisper_model_params": {"audio_ctx": 1024},
        "whisper_transcription_params": {"language": "en"},
    }
    config_file = tmp_config_dir / WHISPER_CONFIG_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    manager = WhisperConfigManager(tmp_config_dir)
    config = manager.get_config()
    assert config.whisper_model_params.audio_ctx == 1024
    assert config.whisper_transcription_params.language == "en"


def test_update_whisper_params_and_save(tmp_config_dir):
    manager = WhisperConfigManager(tmp_config_dir)
    params = WhisperModelParams(language="fr")
    manager.update_whisper_params(params)
    manager.reload()
    assert manager.get_whisper_params().language == "fr"


def test_update_transcription_params_and_save(tmp_config_dir):
    manager = WhisperConfigManager(tmp_config_dir)
    params = WhisperTranscriptionParams(language="fr")
    manager.update_transcription_params(params)
    manager.reload()
    assert manager.get_transcription_params().language == "fr"


def test_set_param_valid_and_invalid(tmp_config_dir):
    manager = WhisperConfigManager(tmp_config_dir)
    manager.set_param("whisper_model_params", "use_gpu", False)
    assert not manager.get_whisper_params().use_gpu
    manager.set_param("whisper_transcription_params", "language", "de")
    assert manager.get_transcription_params().language == "de"
    with pytest.raises(KeyError):
        manager.set_param("invalid_section", "foo", "bar")
    with pytest.raises(KeyError):
        manager.set_param("whisper_model_params", "not_a_key", "bar")


def test_save_and_reload(tmp_config_dir):
    manager = WhisperConfigManager(tmp_config_dir)
    manager.get_whisper_params().language = "en"
    manager._save()
    manager.reload()
    assert manager.get_whisper_params().language == "en"


def test_load_or_initialize_handles_corrupt_yaml(tmp_config_dir):
    config_file = tmp_config_dir / WHISPER_CONFIG_FILENAME
    config_file.write_text("not: valid: yaml: [")
    with mock.patch("builtins.print") as mock_print:
        manager = WhisperConfigManager(tmp_config_dir)
        assert "Failed to load YAML config" in "".join(
            str(a) for a in mock_print.call_args_list
        )
        assert config_file.exists()
        assert isinstance(manager.get_config(), WhisperConfig)
