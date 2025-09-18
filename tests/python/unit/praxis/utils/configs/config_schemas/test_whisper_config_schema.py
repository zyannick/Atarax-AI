import pytest

from ataraxai.praxis.utils.configs.config_schemas.whisper_config_schema import (
    WhisperConfig,
    WhisperModelParams,
    WhisperTranscriptionParams,
)


def test_whisper_model_params_defaults():
    params = WhisperModelParams()
    assert params.n_threads == 0
    assert params.use_gpu is True
    assert params.flash_attn is True
    assert params.audio_ctx == 0
    assert params.model == "data/last_models/models/whisper/ggml-base.en.bin"
    assert params.language == "en"


def test_whisper_model_params_custom_values():
    params = WhisperModelParams(
        n_threads=4,
        use_gpu=False,
        flash_attn=False,
        audio_ctx=128,
        model="custom/model.bin",
        language="fr",
    )
    assert params.n_threads == 4
    assert params.use_gpu is False
    assert params.flash_attn is False
    assert params.audio_ctx == 128
    assert params.model == "custom/model.bin"
    assert params.language == "fr"


def test_whisper_transcription_params_defaults():
    params = WhisperTranscriptionParams()
    assert params.config_version == "1.0"
    assert params.n_threads == 0
    assert params.language == "en"
    assert params.translate is False
    assert params.print_special is False
    assert params.print_progress is True
    assert params.no_context is True
    assert params.max_len == 512
    assert params.single_segment is False
    assert params.temperature == 0.8


def test_whisper_transcription_params_custom_values():
    params = WhisperTranscriptionParams(
        config_version="2.0",
        n_threads=8,
        language="es",
        translate=True,
        print_special=True,
        print_progress=False,
        no_context=False,
        max_len=1024,
        single_segment=True,
        temperature=0.5,
    )
    assert params.config_version == "2.0"
    assert params.n_threads == 8
    assert params.language == "es"
    assert params.translate is True
    assert params.print_special is True
    assert params.print_progress is False
    assert params.no_context is False
    assert params.max_len == 1024
    assert params.single_segment is True
    assert params.temperature == 0.5


def test_whisper_config_defaults():
    config = WhisperConfig()
    assert config.config_version == "1.0"
    assert isinstance(config.whisper_model_params, WhisperModelParams)
    assert isinstance(config.whisper_transcription_params, WhisperTranscriptionParams)


def test_whisper_config_custom_values():
    model_params = WhisperModelParams(n_threads=2, language="de")
    transcription_params = WhisperTranscriptionParams(language="de", temperature=0.3)
    config = WhisperConfig(
        config_version="2.1",
        whisper_model_params=model_params,
        whisper_transcription_params=transcription_params,
    )
    assert config.config_version == "2.1"
    assert config.whisper_model_params.n_threads == 2
    assert config.whisper_model_params.language == "de"
    assert config.whisper_transcription_params.language == "de"
    assert config.whisper_transcription_params.temperature == 0.3
